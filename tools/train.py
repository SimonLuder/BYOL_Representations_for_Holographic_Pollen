import os
import shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse
from pollen_datasets.poleno import PairwiseHolographyImageFolder

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from byol_poleno.utils import config
from byol_poleno.utils.wandb_utils import WandbManager
from byol_poleno.model import BYOLWithTwoImages
from byol_poleno.model.backbones import get_backbone, set_single_channel_input, update_linear_layer


def set_bn_eval(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()


def set_bn_train(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()


class Trainer:

    def __init__(self, model, optimizer, dataloader_train, dataloader_val=None, device="cpu", wandb_run=None, checkpoint_dir=None, val_step=None, accum_steps=1):
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
        self.accum_steps = accum_steps


    def train_one_epoch(self):

        self.model.train()
        self.model.apply(set_bn_eval)  # override: freeze BN
        self.optimizer.zero_grad()
        n_samples_train = []
        all_losses_train = []
        
        pbar = tqdm(self.dataloader_train, desc="Training")
        for ((im1, im2), _, _) in pbar:

            self._step += 1
            
            im1, im2 = im1.to(self.device), im2.to(self.device)

            loss = self.model(x1=im1, x2=im2)
            loss = loss / self.accum_steps
            loss.backward()

            # accumulate gradients
            if (self._step) % self.accum_steps == 0:

                # Enable BN updates for this "true step"
                self.model.apply(set_bn_train)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.update_moving_average()  # Update moving average of target encoder

                # 2. Freeze BN again for accumulation steps
                self.model.apply(set_bn_eval)

            n_samples_train.append(im1.shape[0])
            all_losses_train.append(loss.item())

            # Log batch metrics
            if self.wandb_run is not None:
                self.wandb_run.log(
                    data = {"train_loss":loss.item()},
                    step = self._step
                    )

            if (self.val_step is not None) and (self._step % self.val_step == 0):
                self.validate_one_epoch()
                self.model.train()
                self.model.apply(set_bn_eval)

        epoch_loss = self._compute_epoch_loss(n_samples_train, all_losses_train)

        # Log epoch metrics
        if self.wandb_run is not None:
            self.wandb_run.log(
                data = {"train_epoch_loss": epoch_loss},
                step = self._step
                )

    
    def validate_one_epoch(self):

        n_samples_val = []
        all_losses_val = []
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.dataloader_val, desc="Validation")
            for ((im1, im2), _, _) in pbar:
                im1, im2 = im1.to(self.device), im2.to(self.device)

                loss = self.model(x1=im1, x2=im2)

                n_samples_val.append(im1.shape[0])
                all_losses_val.append(loss.item())

            epoch_loss = self._compute_epoch_loss(n_samples_val, all_losses_val)
            print("epoch loss", epoch_loss)

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
    model_conf = conf['byol']

    # Logging with WandB
    run_name = f"byol_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            [0.5,] * dataset_conf["img_channels"],
            [0.5,] * dataset_conf["img_channels"],
        )
    )
    transform = transforms.Compose(transforms_list)

    # Datasets
    dataset_train = PairwiseHolographyImageFolder(
        root=dataset_conf["root"], 
        transform=transform, 
        dataset_cfg=dataset_conf,
        labels=dataset_conf.get("labels_train"),
        verbose=True
    )
    if dataset_conf.get("labels_val"):
        dataset_val = PairwiseHolographyImageFolder(
            root=dataset_conf["root"], 
            transform=transform, 
            dataset_cfg=dataset_conf,
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
    backbone = get_backbone(model_conf["backbone"], pretrained=False)
    backbone = set_single_channel_input(backbone)
    if model_conf["embedding_size"] is not None:
        backbone = update_linear_layer(backbone, layer=model_conf["hidden_layer"], out_features=model_conf["embedding_size"])

    # BYOL
    model = BYOLWithTwoImages(
        backbone,
        image_size = dataset_conf.get("img_interpolation", dataset_conf["img_size"]),
        image_channels = dataset_conf.get("img_channels", 1),
        hidden_layer=model_conf.get("hidden_layer", "avgpool"),
        projection_size=model_conf.get("projection_size", 256),
        projection_hidden_size=model_conf.get("projection_hidden_size", 4096),
        augment_fn=torch.nn.Identity(),
        augment_fn2=None,
        moving_average_decay=model_conf.get("moving_average_decay", 0.99),
        use_momentum=model_conf.get("use_momentum", True),
        )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf["lr"])

    # Model trainer
    checkpoint_dir = os.path.join("checkpoints", run_name)
    val_step=train_conf["validation_step"]
    acc_steps = train_conf.get("grad_accumulation_steps", 1)
    trainer = Trainer(model, optimizer, dataloader_train, dataloader_val, 
                      device, wandb_run, checkpoint_dir, val_step, acc_steps)
    
    # Save config file to checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    destination = os.path.join(checkpoint_dir, os.path.basename(config_path))
    shutil.copy2(config_path, destination)

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