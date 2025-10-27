import os
import shutil
from datetime import datetime
import argparse
from pollen_datasets.poleno import PairwiseHolographyImageFolder

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from utils import config
from utils.custom_byol import BYOLWithTwoImages
from model.backbones import get_backbone, set_single_channel_input


class BYOLLightningModule(pl.LightningModule):
    def __init__(self, backbone, image_size, image_channels, lr):
        super().__init__()
        self.model = BYOLWithTwoImages(
            backbone,
            image_size=image_size,
            image_channels=image_channels,
            augment_fn=torch.nn.Identity(),
            hidden_layer="avgpool",
            sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        )
        self.lr = lr
        self.best_val_loss = float("inf")

    def training_step(self, batch, batch_idx):
        (im1, im2), _, _ = batch
        loss = self.model(x1=im1, x2=im2)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (im1, im2), _, _ = batch
        loss = self.model(x1=im1, x2=im2)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def on_before_zero_grad(self, _):
        if self.model.use_momentum:
            self.model.update_moving_average()


def main(config_path):

    # Config
    conf = config.load(config_path)
    dataset_conf = conf["dataset"]
    train_conf = conf["training"]
    model_conf = conf["byol"]

    pl.seed_everything(42, workers=True)

    # Logging with WandB
    run_name = f"byol_lightning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(
        project="ByolHolographicPollen",
        name=run_name,
        config=train_conf,
    )

    # Transformations
    transforms_list = [transforms.ToTensor()]
    if dataset_conf.get("img_interpolation"):
        transforms_list.append(
            transforms.Resize(
                (dataset_conf["img_interpolation"], 
                 dataset_conf["img_interpolation"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    transforms_list.append(
        transforms.Normalize(
            (0.5,) * dataset_conf["img_channels"],
            (0.5,) * dataset_conf["img_channels"],
        )
    )

    transform = transforms.Compose(transforms_list)

    # Check if labels fies exist
    if not os.path.isfile(dataset_conf["labels_train"]):
        raise FileNotFoundError(f'File {dataset_conf["labels_train"]} does not exist')
    if not os.path.isfile(dataset_conf["labels_val"]):
        raise FileNotFoundError(f'File {dataset_conf["labels_val"]} does not  exist')

    # Datasets
    dataset_train = PairwiseHolographyImageFolder(
        root=dataset_conf["root"],
        transform=transform,
        config=dataset_conf,
        labels=dataset_conf.get("labels_train"),
        verbose=True,
    )
    dataset_val = None
    if dataset_conf.get("labels_val"):
        dataset_val = PairwiseHolographyImageFolder(
            root=dataset_conf["root"],
            transform=transform,
            config=dataset_conf,
            labels=dataset_conf.get("labels_val"),
            verbose=True,
        )

    # DataLoaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=train_conf["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dataloader_val = None
    if dataset_val:
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=train_conf["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Backbone
    backbone = get_backbone(model_conf["backbone"], pretrained=False)
    backbone = set_single_channel_input(backbone)

    # Lightning model
    model = BYOLLightningModule(
        backbone=backbone,
        image_size=dataset_conf.get("img_interpolation", dataset_conf["img_size"]),
        image_channels=dataset_conf.get("img_channels", 1),
        lr=train_conf["lr"],
    )

    # Callbacks
    checkpoint_dir = os.path.join("checkpoints", run_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    global_batch_size = (
        train_conf["batch_size"]
        * train_conf.get("accumulate_grad_batches", 1)
        * train_conf.get("num_devices", 1)
        * train_conf.get("num_nodes", 1)
        )

    wandb_logger.log_hyperparams({
    **train_conf,
    "global_batch_size": global_batch_size
    })

    # Save config file to checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    destination = os.path.join(checkpoint_dir, os.path.basename(config_path))
    shutil.copy2(config_path, destination)

    print(f"✅ Global batch size: {global_batch_size}")
    print("Num_devices:", torch.cuda.device_count())

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_conf["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        num_nodes=train_conf.get("num_nodes", 1),
        devices=train_conf.get("num_devices", 1),
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=50,
        val_check_interval=train_conf.get("validation_step", 100) * train_conf["accumulate_grad_batches"], # Global steps
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        accumulate_grad_batches=train_conf.get("accumulate_grad_batches", 1),
    )

    # Fit
    trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path of the configuration file.")
    parser.add_argument("--config", default="config/slurm_distributed_config.yaml", type=str)
    args = parser.parse_args()
    main(args.config)