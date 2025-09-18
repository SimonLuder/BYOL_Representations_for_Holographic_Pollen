import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse
from typing import Optional
from pollen_datasets.poleno import PairwiseHolographyImageFolder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models

from utils import config
from utils.wandb_utils import WandbManager
from utils.custom_byol import BYOLWithTwoImages
from utils.model_setup import set_single_channel_input


class Trainer:

    def __init__(self, model, optimizer, dataloader_train, dataloader_val=None, device="cpu", wandb_run=None, checkpoint_dir=None, val_step=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.wandb_run = wandb_run
        self.checkpoint_dir = checkpoint_dir
        self._step = 0
        self.best = 1e9
        self.val_step = val_step
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val


    def train_one_epoch(self):

        self.model.train()
        n_samples = []
        all_losses = []
        
        pbar = tqdm(self.dataloader_train, desc="Training")
        for ((im1, im2), _, _) in pbar:
            im1, im2 = im1.to(self.device), im2.to(self.device)

            loss = self.model(x1=im1, x2=im2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.update_moving_average()  # Update moving average of target encoder

            n_samples.append(im1.shape[0])
            all_losses.append(loss.item())

            # Log batch metrics
            if self.wandb_run is not None:
                self.wandb_run.log(
                    data = {"train_loss":loss.item()},
                    step = self._step
                    )
                self._step += 1

            if (self.val_step is not None) and (self._step % self.val_step == 0):
                self.validate_one_epoch()

        epoch_loss = self._compute_epoch_loss(n_samples, all_losses)

        # Log epoch metrics
        if self.wandb_run is not None:
            self.wandb_run.log(
                data = {"train_epoch_loss": epoch_loss},
                step = self._step
                )

    
    def validate_one_epoch(self):

        n_samples = []
        all_losses = []
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.dataloader_val, desc="Validation")
            for ((im1, im2), _, _) in pbar:
                im1, im2 = im1.to(self.device), im2.to(self.device)

                loss = self.model(x1=im1, x2=im2)

                n_samples.append(im1.shape[0])
                all_losses.append(loss.item())

        epoch_loss = self._compute_epoch_loss(n_samples, all_losses)

        # Log epoch metrics
        if self.wandb_run is not None:
            self.wandb_run.log(
                data = {"val_epoch_loss": epoch_loss},
                step = self._step
                )
            
        if epoch_loss < self.best and self.checkpoint_dir is not None:
            self.create_model_checkpoint(name="best.pth")
    

    def _compute_epoch_loss(self, n_samples, all_losses):
        N = np.array(n_samples)
        L = np.array(all_losses)
        epoch_loss = (L @ N) / N.sum()
        return epoch_loss
    

    def create_model_checkpoint(self, name):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if name is None:
            torch.save(self.model.state_dict(), 
                       os.path.join(self.checkpoint_dir, f"{self._step}.pth"))
        else:
            torch.save(self.model.state_dict(), 
                       os.path.join(self.checkpoint_dir, name))
    

def main(config_path):

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Configuration
    conf = config.load(config_path)
    dataset_conf = conf['dataset']
    train_conf = conf['training']

    # Logging with WandB
    run_name = f"byol_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_manager = WandbManager(
        project="ByolHolographicPollen", 
        run_name=run_name, 
        config=train_conf
    )
    wandb_run = wandb_manager.get_run()

    # Transformations
    transforms_list = []
    transforms_list.append(transforms.ToTensor())

    if dataset_conf.get("img_interpolation"):
        transforms_list.append(
            transforms.Resize(
                (dataset_conf["img_interpolation"], 
                 dataset_conf["img_interpolation"]),
                 interpolation = transforms.InterpolationMode.BILINEAR
            )
        )
    transforms_list.append(
        transforms.Normalize(
            (0.5) * dataset_conf["img_channels"], 
            (0.5) * dataset_conf["img_channels"]
        )
    )
    transform = transforms.Compose(transforms_list)

    # Datasets
    dataset_train = PairwiseHolographyImageFolder(
        root=dataset_conf["root"], 
        transform=transform, 
        config=dataset_conf,
        labels=dataset_conf.get("labels_train"),
        verbose=True
    )
    if dataset_conf.get("labels_val"):
        dataset_val = PairwiseHolographyImageFolder(
            root=dataset_conf["root"], 
            transform=transform, 
            config=dataset_conf,
            labels=dataset_conf.get("labels_val"),
            verbose=True)
    
    # Dataloaders
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=train_conf["batch_size"],
                                  shuffle=True)
    
    if dataset_conf.get("labels_val"):
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=train_conf["batch_size"],
                                    shuffle=False)

    # Backbone
    backbone = models.resnet50(pretrained=True)
    backbone = set_single_channel_input(backbone)

    # BYOL
    model = BYOLWithTwoImages(
        backbone,
        image_size = dataset_conf.get("img_interpolation", dataset_conf["img_size"]),
        image_channels = dataset_conf.get("img_channels", 1),
        augment_fn=torch.nn.Identity(), # No augmentation with 2 different images and grayscale
        hidden_layer = 'avgpool')
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf["lr"])

    # Model trainer
    checkpoint_dir = os.path.join("models", run_name)
    val_step=train_conf["validation_step"]
    trainer = Trainer(model, optimizer, dataloader_train, dataloader_val, 
                      device, wandb_run, checkpoint_dir, val_step)

    for epoch_idx in range(train_conf["num_epochs"]):
        
        # Train one epoch
        train_logs = trainer.train_one_epoch()
        print(train_logs)

        # Validate one epoch
        if dataset_conf.get("labels_val"):
            val_logs = trainer.validate_one_epoch()
            print(val_logs)
            
    # Save checkpoint
    trainer.create_model_checkpoint(name="{run_name}.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path of the configuration file.')
    parser.add_argument('--config', default='config/base_config.yaml', type=str)
    args = parser.parse_args()

    main(args.config)