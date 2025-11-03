import os
import numpy as np
import argparse
from tqdm import tqdm

import torch
from tools.train import BYOLWithTwoImages
from torch.utils.data import DataLoader
from torchvision import transforms
from pollen_datasets.poleno import PairwiseHolographyImageFolder

from model.backbones import get_backbone, set_single_channel_input, update_linear_layer
from utils import config
from utils.config import get_ckpt_config_file


def load_model_weights(model, ckpt_path):
    """
    Load weights from either:
      - a Lightning checkpoint (.ckpt)
      - a plain PyTorch model (.pth)
    """

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Detect Lightning checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        print(f"[INFO] Detected Lightning checkpoint: {ckpt_path}")
        lightning_state_dict = ckpt["state_dict"]
        state_dict = {
            k.replace("model.", "", 1): v for k, v in lightning_state_dict.items()
            }
    # Detect standard PyTorch state_dict
    elif isinstance(ckpt, dict):
        print(f"[INFO] Detected standard PyTorch state_dict: {ckpt_path}")
        state_dict = ckpt
    # Detect model object (rare, but supported)
    else:
        print(f"[INFO] Detected full model object in checkpoint.")
        model = ckpt
        return model

    model.load_state_dict(state_dict, strict=False)
    print("[INFO] Model weights successfully loaded.")
    return model


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def inference(ckpt_path, config_path, config_updates=None):

    conf = config.load(config_path)
    if config_updates:
        conf = deep_update(conf, config_updates)
    dataset_conf = conf["dataset"]
    model_conf = conf ["byol"]

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

    print(dataset_conf.get("labels_test"))
    print(os.path.exists(dataset_conf.get("labels_test")))

    # Dataset
    dataset_test = PairwiseHolographyImageFolder(
        root=dataset_conf["root"],
        transform=transform,
        config=dataset_conf,
        labels=dataset_conf.get("labels_test"),
        verbose=True,
    )

    # DataLoader
    dataloader = DataLoader(
        dataset_test,
        batch_size=4,
        shuffle=False,
        num_workers=4,
    )

    # ----------------------------
    # 2. Rebuild BYOL model
    # ----------------------------

    # Backbone
    backbone = get_backbone(model_conf["backbone"], pretrained=False)
    backbone = set_single_channel_input(backbone)
    backbone = update_linear_layer(
        backbone, 
        layer=model_conf["hidden_layer"], 
        out_features=model_conf["embedding_dim"]
    )

    # Recreate BYOL model
    model = BYOLWithTwoImages(
        backbone,
        image_size=224,    # <- same as dataset_conf["img_interpolation"] or img_size
        image_channels=dataset_conf["img_channels"], 
        augment_fn=torch.nn.Identity(),
        hidden_layer="avgpool"
    )

    # ----------------------------
    # 3. Load weights (fix Lightning prefixes)
    # ----------------------------
    model = load_model_weights(model, ckpt_path)
    model.eval()

    all_proj1, all_proj2 = [], []
    all_emb1, all_emb2 = [], []
    all_filenames1, all_filenames2 = [], []

    for (images, _, filenames) in tqdm(dataloader):
        img1, img2 = images
        file1, file2 = filenames
        
        with torch.no_grad():
            ((proj1, emb1), (proj2, emb2)) = model(img1, img2, return_embedding=True, return_projection=True)

            # Move to CPU and convert to numpy
            all_proj1.append(proj1.cpu().numpy())
            all_proj2.append(proj2.cpu().numpy())
            all_emb1.append(emb1.cpu().numpy())
            all_emb2.append(emb2.cpu().numpy())
            all_filenames1.extend(file1)
            all_filenames2.extend(file2)

    # Stack everything into single arrays
    all_proj1 = np.concatenate(all_proj1, axis=0)
    all_proj2 = np.concatenate(all_proj2, axis=0)
    all_emb1 = np.concatenate(all_emb1, axis=0)
    all_emb2 = np.concatenate(all_emb2, axis=0)
    all_filenames1 = np.array(all_filenames1)
    all_filenames2 = np.array(all_filenames2)

    # Save in a compressed npz file
    save_dir = os.path.split(ckpt_path)[0]
    filename = os.path.join(save_dir, "embeddings.npz")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    np.savez_compressed(
        filename,
        proj1=all_proj1,
        proj2=all_proj2,
        emb1=all_emb1,
        emb2=all_emb2,
        files1=all_filenames1,
        files2=all_filenames2
    )


if __name__ == "__main__":

    ckpt_file = "checkpoints/byol_lightning_20251030_164712/last.ckpt"
    ckpt_file = "checkpoints/byol_lightning_20251030_164712/epoch_epoch=49-step=3088-val_loss=0.0524.ckpt"
    config_file = None

    parser = argparse.ArgumentParser(description='Path of the configuration file.')
    parser.add_argument('--ckpt', default=ckpt_file, type=str)
    parser.add_argument('--config', default=config_file, type=str)
    args = parser.parse_args()

    if args.config is None:
        args.config = get_ckpt_config_file(args.ckpt)

    print(f"Running inference with checkpoint: {args.ckpt} and config: {args.config}")

    config_updates = {
        "dataset": {
            "root": "Z:/marvel/marvel-fhnw/data/Poleno/",
            "labels_train": "data/processed/poleno/labels_train.csv",
            "labels_val": "data/processed/poleno/labels_val.csv",
            "labels_test": "data/processed/poleno/labels_test.csv",
        }
    }

    inference(args.ckpt, args.config, config_updates=config_updates)