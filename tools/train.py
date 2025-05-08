import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models

import numpy as np
from datetime import datetime
from tqdm import tqdm
from pollen_datasets.poleno import PairwiseHolographyImageFolder

from utils import config
from utils.wandb_utils import WandbManager
from utils.custom_byol import BYOLWithTwoImages


def set_single_channel_input(model, layer_name=None):
    """
    Update the model to accept single-channel (grayscale) input images.
    
    Parameters:
        model (nn.Module): The model to modify.
        layer_name (str, optional): Name of the layer to replace. If None, will auto-detect.
        
    Returns:
        nn.Module: Updated model.
    """
    def convert_conv(conv):
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode
        )
        with torch.no_grad():
            new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
            if conv.bias is not None:
                new_conv.bias[:] = conv.bias
        return new_conv

    if layer_name:
        # Directly replace a named layer
        parts = layer_name.split(".")
        mod = model
        for p in parts[:-1]:
            mod = getattr(mod, p)
        setattr(mod, parts[-1], convert_conv(getattr(mod, parts[-1])))
    else:
        # Auto-detect and replace the first 3-channel conv layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                # Navigate to the parent module and replace
                parent = model
                subnames = name.split(".")
                for sub in subnames[:-1]:
                    parent = getattr(parent, sub)
                setattr(parent, subnames[-1], convert_conv(module))
                break

    return model


class Trainer:
    def __init__(self, model, optimizer, device, wandb_run=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.wandb_run = wandb_run
        self._step = 0


    def train_one_epoch(self, dataloader):

        self.model.train()
        n_samples = []
        all_losses = []
        
        pbar = tqdm(dataloader, desc="Training")
        for ((im1, im2), _, _) in pbar:
            im1, im2 = im1.to(self.device), im2.to(self.device)

            loss = self.model(x1=im1, x2=im2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            n_samples.append(im1.shape[0])
            all_losses.append(loss.item())

            # Log batch metrics
            if self.wandb_run is not None:
                self.wandb_run.log(
                    data = {"train_loss":loss.item()},
                    step = self._step
                    )
                self._step += 1

        logs = self._compute_epoch_loss()

        # Log epoch metrics
        if self.wandb_run is not None:
            self.wandb_run.log(
                data = {"train_epoch_loss":logs},
                step = self._step
                )

    
    def validate_one_epoch(self, dataloader):

        self.model.eval()
        n_samples = []
        all_losses = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for ((im1, im2), _, _) in pbar:
                im1, im2 = im1.to(self.device), im2.to(self.device)

                loss = self.model(x1=im1, x2=im2)

                n_samples.append(im1.shape[0])
                all_losses.append(loss.item())

        logs = self._compute_epoch_loss(n_samples, all_losses)

        # Log epoch metrics
        if self.wandb_run is not None:
            self.wandb_run.log(
                data = {"val_epoch_loss":logs},
                step = self._step
                )
    

    def _compute_epoch_loss(self, n_samples, all_losses):
        N = np.array(n_samples)
        L = np.array(all_losses)
        epoch_loss = (L @ N) / N.sum()
        return {"epoch_loss": epoch_loss}


def main(config_path):

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuration
    conf = config.load(config_path)
    dataset_conf = conf['dataset']
    train_conf = conf['training']

    # Logging with WandB
    run_name = f"byol_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_manager = WandbManager(project="ByolHolographicPollen", 
                                 run_name=run_name, 
                                 config=train_conf)
    wandb_run = wandb_manager.get_run()

    # Transformations
    transforms_list = []
    transforms_list.append(transforms.ToTensor())

    if dataset_conf.get("img_interpolation"):
        transforms_list.append(
            transforms.Resize((dataset_conf["img_interpolation"], 
                               dataset_conf["img_interpolation"]),
                               interpolation = transforms.InterpolationMode.BILINEAR))
        
    transforms_list.append(
        transforms.Normalize(
            (0.5) * dataset_conf["img_channels"], 
            (0.5) * dataset_conf["img_channels"]))

    transform = transforms.Compose(transforms_list)

    # Datasets
    dataset_train = PairwiseHolographyImageFolder(
        root=dataset_conf["root"], 
        transform=transform, 
        config=dataset_conf,
        labels=dataset_conf.get("labels_train"),
        verbose=True)
    

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
    trainer = Trainer(model, optimizer, device, wandb_run)


    for epoch_idx in range(train_conf["num_epochs"]):
        
        train_logs = trainer.train_one_epoch(model, optimizer, dataloader_train, device)
        print(train_logs)

        if dataset_conf.get("labels_val"):
            val_logs = trainer.validate_one_epoch(model, dataloader_val, device)


if __name__ == "__main__":

    config_path = "config/base_config.yaml"

    main(config_path)