import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import clip

from pollen_datasets.poleno import PairwiseHolographyImageFolder
from byol_poleno.utils import config


# CLIP image encoder wrapper (official repo)
class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        feats = self.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


# Inference
def inference(
    conf,
    clip_model="ViT-B/32",
    save_dir="checkpoints/clip_vision/inference/baseline/",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_conf = conf["dataset"]

    # Transforms (official CLIP preprocessing)
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    img_size = dataset_conf.get("img_interpolation", 224)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            img_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # [3,H,W]
        transforms.Normalize(clip_mean, clip_std),
    ])

    # Dataset
    dataset_test = PairwiseHolographyImageFolder(
        root=dataset_conf["root"],
        transform=transform,
        dataset_cfg=dataset_conf,
        labels=dataset_conf.get("labels_test"),
        verbose=True,
    )

    dataloader = DataLoader(
        dataset_test,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
    )

    # CLIP model
    model = CLIPImageEncoder(
        model_name=clip_model,
        device=device,
    ).to(device)

    model.eval()

    # Inference loop
    all_emb1, all_emb2 = [], []
    all_filenames1, all_filenames2 = [], []

    for (images, _, filenames) in tqdm(dataloader):
        img1, img2 = images
        file1, file2 = filenames

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        with torch.no_grad():
            emb1 = model(img1)
            emb2 = model(img2)

        all_emb1.append(emb1.cpu().numpy())
        all_emb2.append(emb2.cpu().numpy())
        all_filenames1.extend(file1)
        all_filenames2.extend(file2)

    # Save
    all_emb1 = np.concatenate(all_emb1, axis=0)
    all_emb2 = np.concatenate(all_emb2, axis=0)

    all_filenames1 = np.array(all_filenames1)
    all_filenames2 = np.array(all_filenames2)

    os.makedirs(save_dir, exist_ok=True)
    save_as = os.path.splitext(os.path.basename(dataset_conf.get("labels_test")))[0]
    filename = os.path.join(save_dir, f"inference_{save_as}.npz")

    dataset_file = os.path.basename(dataset_conf.get("labels_test"))

    np.savez_compressed(
        filename,
        dataset=dataset_file,
        emb1=all_emb1,
        emb2=all_emb2,
        files1=all_filenames1,
        files2=all_filenames2,
    )

    print(f"[INFO] Saved CLIP embeddings to: {filename}")


# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CLIP inference on paired images")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    args = parser.parse_args()

    config_updates = {
        "dataset": {
            "root": "Z:/marvel/marvel-fhnw/data/",
            "labels_test": "data/final/poleno/basic_test_20.csv",
        }
    }

    conf = config.load(args.config)
    conf = config.deep_update(conf, config_updates)

    inference(
        conf=conf,
        clip_model=args.clip_model,
    )

    config_updates = {
        "dataset": {
            "root": "Z:/marvel/marvel-fhnw/data/",
            "labels_test": "data/final/poleno/isolated_test_20.csv",
        }
    }

    conf = config.load(args.config)
    conf = config.deep_update(conf, config_updates)

    inference(
        conf=conf,
        clip_model=args.clip_model,
    )

# python -m tools.baselines.clip_vision_encoding --config config/base_distributed_config.yaml