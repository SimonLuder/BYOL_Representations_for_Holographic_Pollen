import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from pollen_datasets.poleno import PairwiseHolographyImageFolder
from byol_poleno.utils import config


# DINOv2 image encoder wrapper
class DINOv2ImageEncoder(torch.nn.Module):
    def __init__(self, model_name="dinov2_vitb14", device="cpu"):
        super().__init__()

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        feats = self.model(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def inference(
    conf,
    dino_model="dinov2_vitb14",
    save_dir="checkpoints/dinov2_vision/inference/baseline/",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_conf = conf["dataset"]

    # DINOv3 preprocessing (ImageNet style)
    img_size = dataset_conf.get("img_interpolation", 224)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            img_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale → RGB
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    # Dataset & loader
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

    # Model
    model = DINOv2ImageEncoder(
        model_name=dino_model,
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

    # Save results
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

    print(f"[INFO] Saved DINOv3 embeddings to: {filename}")


# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DINOv3 inference on paired images"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--dino_model",
        type=str,
        default="dinov2_vitb14",
        help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14",
    )
    args = parser.parse_args()

    # First dataset
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
        dino_model=args.dino_model,
    )

    # Second dataset
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
        dino_model=args.dino_model,
    )


# python -m tools.baselines.dinov3_encoding --config config/base_distributed_config.yaml