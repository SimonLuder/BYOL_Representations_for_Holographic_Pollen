import numpy as np
import torch
import torch.nn as nn
from tools.train_lightning import BYOLLightningModule
from tools.train import BYOLWithTwoImages
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from utils.model_setup import set_single_channel_input
from utils import config
from pollen_datasets.poleno import PairwiseHolographyImageFolder


def main():
    config_path = "config/base_config.yaml"

    # Path to your checkpoint
    ckpt_path = r"C:\Users\simon\Downloads\byol_lightning_20250925_110043\epoch=3-step=938-val_loss=0.0849.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    conf = config.load(config_path)
    dataset_conf = conf["dataset"]

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
    backbone = models.resnet50(weights=None)
    backbone = set_single_channel_input(backbone)

    # Recreate BYOL model
    model = BYOLWithTwoImages(
        backbone,
        image_size=224,    # <- same as dataset_conf["img_interpolation"] or img_size
        image_channels=1,  # <- same as dataset_conf["img_channels"]
        augment_fn=torch.nn.Identity(),
        hidden_layer="avgpool"
    )

    # ----------------------------
    # 3. Load weights (fix Lightning prefixes)
    # ----------------------------
    lightning_state_dict = ckpt["state_dict"]
    state_dict = {k.replace("model.", "", 1): v for k, v in lightning_state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()


    all_proj1, all_proj2 = [], []
    all_emb1, all_emb2 = [], []
    all_filenames1, all_filenames2 = [], []

    for (images, _, filenames) in dataloader:
        img1, img2 = images
        file1, file2 = filenames
        
        with torch.no_grad():
            ((proj1, emb1), (proj2, emb2)) = model(img1, img2, return_embedding=True, return_projection=True)

            cos = nn.CosineSimilarity(dim=1)
            emb_sim = cos(emb1, emb2)
            print("Cosine similarity between embeddings:", emb_sim)
            proj_sim = cos(proj1, proj2)
            print("Cosine similarity between projections:", proj_sim)

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
    np.savez_compressed(
        "embeddings_and_projections.npz",
        proj1=all_proj1,
        proj2=all_proj2,
        emb1=all_emb1,
        emb2=all_emb2,
        files1=all_filenames1,
        files2=all_filenames2
    )


if __name__ == "__main__":
    main()