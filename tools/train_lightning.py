import os
import shutil
import argparse
from datetime import datetime, timedelta
from pollen_datasets.poleno import HolographyImageFolder, PairwiseHolographyImageFolder

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from ssl_poleno.utils import config
from ssl_poleno.model.lightning import LITSSLModel
from ssl_poleno.model.objectives import NormalizedL2Objective, VICRegObjective, HybridObjective, Hybrid2Objective
from ssl_poleno.model.backbones import get_backbone, set_single_channel_input, update_linear_layer
        

def main(config_path):

    # Config
    conf = config.load(config_path)
    dataset_conf = conf["dataset"]
    transform_conf = conf.get("transforms", {})
    train_conf = conf["training"]
    model_conf = conf["ssl"]
    cond_conf = conf.get("conditioning", {})

    pl.seed_everything(42, workers=True)
   
    # Logging with WandB
    conf_flat = {}
    for section_name, section_conf in conf.items():
        if isinstance(section_conf, dict):
            for k, v in section_conf.items():
                conf_flat[f"{section_name}.{k}"] = v
        else:
            conf_flat[section_name] = section_conf

    run_name = f"{model_conf['objective']}_lit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(
        project="ByolHolographicPollen",
        name=run_name,
        config=conf_flat,
    )
    checkpoint_dir = os.path.join("checkpoints", run_name)

    # Save config file to checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    destination = os.path.join(checkpoint_dir, os.path.basename(config_path))
    shutil.copy2(config_path, destination)

    def get_transform(transform_conf, img_size, img_channels):
        
        # Transformations
        transforms_list = [transforms.ToTensor()]

        if transform_conf.get("img_interpolation"):
            print("Using image interpolation:", transform_conf["img_interpolation"])
            transforms_list.append(
                transforms.Resize(
                    (transform_conf["img_interpolation"], 
                     transform_conf["img_interpolation"]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )
        transforms_list.append(
            transforms.Normalize(
                [0.5,] * img_channels,
                [0.5,] * img_channels,
            )
        )

        gaussian_blur = transform_conf.get("gaussian_blur", {"enabled": False})
        if gaussian_blur.get("enabled", False):
            print("Using GaussianBlur augmentation")
            transforms_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(
                        gaussian_blur.get("kernel_size", (3, 3)),
                        gaussian_blur.get("sigma", (1.0, 2.0)),
                    )],
                    p=gaussian_blur.get("p", 0.2)
                )
            )

        rrc = transform_conf.get("random_resized_crop", {"enabled": False})
        if rrc.get("enabled", False):
            print("Using RandomResizedCrop augmentation")
            transforms_list.append(
                transforms.RandomApply(
                    [transforms.RandomResizedCrop(
                        size=dataset_conf.get("img_interpolation", img_size),
                        scale=rrc.get("scale", (0.9, 1.0)),
                        ratio=rrc.get("ratio", (1.0, 1.0)),
                    )],
                p=rrc.get("p", 0.5)
                )
            )

        return transforms.Compose(transforms_list)
    
    transform = get_transform(transform_conf, dataset_conf.get("img_size", 1), dataset_conf.get("img_channels", 1))

    print(transform)

    # Check if labels fies exist
    if not os.path.isfile(dataset_conf["labels_train"]):
        raise FileNotFoundError(f'File {dataset_conf["labels_train"]} does not exist')
    if not os.path.isfile(dataset_conf["labels_val"]):
        raise FileNotFoundError(f'File {dataset_conf["labels_val"]} does not exist')
    
    # Datasets
    dataset = PairwiseHolographyImageFolder if dataset_conf["paired_images"] else HolographyImageFolder
    
    dataset_train = dataset(
        root=dataset_conf["root"],
        transform=transform,
        dataset_cfg=dataset_conf,
        labels=dataset_conf.get("labels_train"),
        verbose=True,
    )

    dataset_val = None
    if dataset_conf.get("labels_val"):
        dataset_val = dataset(
            root=dataset_conf["root"],
            transform=transform,
            dataset_cfg=dataset_conf,
            cond_cfg=cond_conf,
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
    if model_conf["embedding_size"] is not None:
        backbone = update_linear_layer(backbone, layer=model_conf["hidden_layer"], out_features=model_conf["embedding_size"])

    # Objective
    objective_name = str(model_conf["objective"]).lower()

    if objective_name in ["byol", "simsiam"]:
        objective = NormalizedL2Objective()
        use_prediction_head = True

    elif objective_name == "vicreg":
        objective = VICRegObjective(
            lambda_inv=model_conf.get("lambda_inv", 1),
            lambda_var=model_conf.get("lambda_var", 1),
            lambda_cov=model_conf.get("lambda_cov", 0.04))
        use_prediction_head=False

    elif objective_name == "hybrid":
        objective = HybridObjective(
            lambda_inv=model_conf.get("lambda_inv", 1),
            lambda_var=model_conf.get("lambda_var", 1),
            lambda_cov=model_conf.get("lambda_cov", 0.04))
        use_prediction_head=False
        
    else: 
        raise ValueError(
        f'Invalid objective "{objective_name}". '
        'Choose from ["byol", "simsiam", "vicreg", "hybrid", "hybrid2"].'
        )

    # Lightning model
    model = LITSSLModel(
        backbone=backbone,
        objective=objective,
        image_size=dataset_conf.get("img_interpolation", dataset_conf["img_size"]),
        image_channels=dataset_conf.get("img_channels", 1),
        hidden_layer=model_conf.get("hidden_layer", "avgpool"),
        projection_size=model_conf.get("projection_size", 256),
        projection_hidden_size=model_conf.get("projection_hidden_size", 4096),
        augment_fn=torch.nn.Identity(), # TODO Update for single image training
        augment_fn2=None,
        moving_average_decay=model_conf.get("moving_average_decay", 0.99),
        use_momentum=model_conf.get("use_momentum", True),
        lr=train_conf["lr"],
        val_knn=model_conf.get("val_knn", False),
        use_prediction_head=use_prediction_head,
    )

    # Callbacks
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_{epoch}-{step}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    knn_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_knn_{epoch}-{step}-{val_cls_knn_acc_epoch:.4f}",
        monitor="val_cls_knn_acc_epoch",
        mode="max",                 
        save_top_k=1,
        save_last=False,
    )

    mrr_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_mrr_{epoch}-{step}-{val_event_mrr_epoch:.4f}",
        monitor="val_event_mrr_epoch",
        mode="max",                 
        save_top_k=1,
        save_last=False,
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

    print(f"Global batch size: {global_batch_size}")
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
        callbacks=[best_checkpoint_callback, knn_checkpoint_callback, mrr_checkpoint_callback],
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