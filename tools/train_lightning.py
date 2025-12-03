import os
import shutil
from datetime import datetime, timedelta
import argparse
from pollen_datasets.poleno import PairwiseHolographyImageFolder

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from byol_poleno.utils import config
from byol_poleno.model import BYOLWithTwoImages
from byol_poleno.model.backbones import get_backbone, set_single_channel_input, update_linear_layer


class BYOLLightningModule(pl.LightningModule):
    def __init__(self, backbone, image_size, image_channels, hidden_layer="avgpool", projection_size=256, projection_hidden_size=4096, 
                 augment_fn=torch.nn.Identity(), augment_fn2=None, moving_average_decay=0.99, use_momentum=True, lr=3e-4):
        super().__init__()
        self.model = BYOLWithTwoImages(
            backbone,
            image_size=image_size,
            image_channels=image_channels,
            hidden_layer=hidden_layer,
            projection_size=projection_size,
            projection_hidden_size=projection_hidden_size,
            augment_fn=augment_fn,
            augment_fn2 = augment_fn2, 
            moving_average_decay=moving_average_decay,
            use_momentum=use_momentum,
            sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        )
        self.lr = lr
        self.best_val_loss = float("inf")

    def training_step(self, batch, batch_idx):
        (im1, im2), _, _ = batch
        loss = self.model(x1=im1, x2=im2)

        local_bs = im1.size(0)

        # Log loss and training samples seen
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=local_bs)
        return loss

    def validation_step(self, batch, batch_idx):
        (im1, im2), _, _ = batch
        loss = self.model(x1=im1, x2=im2)

        local_bs = im1.size(0)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=local_bs)
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
    conf_flat = {}
    for section_name, section_conf in conf.items():
        if isinstance(section_conf, dict):
            for k, v in section_conf.items():
                conf_flat[f"{section_name}.{k}"] = v
        else:
            conf_flat[section_name] = section_conf

    run_name = f"byol_lightning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(
        project="ByolHolographicPollen",
        name=run_name,
        config=conf_flat,
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
        dataset_cfg=dataset_conf,
        labels=dataset_conf.get("labels_train"),
        verbose=True,
    )
    dataset_val = None
    if dataset_conf.get("labels_val"):
        dataset_val = PairwiseHolographyImageFolder(
            root=dataset_conf["root"],
            transform=transform,
            dataset_cfg=dataset_conf,
            labels=dataset_conf.get("labels_val"),
            verbose=True,
        )

    # DataLoaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=train_conf["batch_size"],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = None
    if dataset_val:
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=train_conf["batch_size"],
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    # Backbone
    backbone = get_backbone(model_conf["backbone"], pretrained=model_conf["pretrained"])
    backbone = set_single_channel_input(backbone)
    backbone = update_linear_layer(backbone, layer=model_conf["hidden_layer"], out_features=model_conf["embedding_size"])

    # Lightning model
    model = BYOLLightningModule(
        backbone=backbone,
        image_size=dataset_conf.get("img_interpolation", dataset_conf["img_size"]),
        image_channels=dataset_conf.get("img_channels", 1),
        hidden_layer=model_conf.get("hidden_layer", "avgpool"),
        projection_size=model_conf.get("projection_size", 256),
        projection_hidden_size=model_conf.get("projection_hidden_size", 4096),
        augment_fn=torch.nn.Identity(),
        augment_fn2=None,
        moving_average_decay=model_conf.get("moving_average_decay", 0.99),
        use_momentum=model_conf.get("use_momentum", True),
        lr=train_conf["lr"],
    )

    # Callbacks
    checkpoint_dir = os.path.join("checkpoints", run_name)
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_{epoch}-{step}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    epoch_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch}-{step}-{val_loss:.4f}",
        save_top_k=-1,               
        every_n_epochs=10,            
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

    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(minutes=10),
        )
    else:
        strategy = "auto"

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_conf["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        num_nodes=train_conf.get("num_nodes", 1),
        devices=train_conf.get("num_devices", 1),
        callbacks=[best_checkpoint_callback, epoch_checkpoint_callback],
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
    parser.add_argument("--config", default="config/base_distributed_config.yaml", type=str)
    args = parser.parse_args()
    main(args.config)