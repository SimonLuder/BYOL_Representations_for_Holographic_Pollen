import subprocess
import torch
from pathlib import Path
import os
import re


def scp_download_model_folder(source: str, target: str):
    """
    Copy a folder from source to target using scp.
    """
    result = subprocess.run(
        ["scp", "-r", source, target],
        capture_output=True,
        text=True
)
    if result.returncode == 0:
        print(f"✅ Downloaded {source}")
    else:
        print(f"❌ Failed to download from {source}")
        print(result.stderr)
    return result.returncode


def extract_backbone_state_dict_from_lightning_ckpt(
    ckpt_path: str,
    save_path: str = None,
    backbone_key: str = "model.online_encoder.net",
    map_location: str = "cpu"
):
    """
    Extracts the backbone weights from a PyTorch Lightning BYOL checkpoint.
    
    Args:
        ckpt_path (str): Local path to the .ckpt file.
        save_path (str, optional): Where to save the extracted backbone state_dict (.pt file). 
                                   If None, won't save, just returns the state_dict.
        backbone_key (str): Key prefix in checkpoint for the backbone model weights.
                            For BYOL, typically "model.online_encoder.net".
        map_location (str): Where to map tensors ('cpu' or 'cuda').

    Returns:
        dict: Cleaned backbone state_dict (keys ready for torch backbone load).
    """

    # Load Lightning checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # Try to find model weights
    state_dict = ckpt.get("state_dict", ckpt)  # works for Lightning and plain torch saves

    # Extract backbone weights
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(backbone_key):
            new_key = k.replace(f"{backbone_key}.", "")
            backbone_state_dict[new_key] = v

    if not backbone_state_dict:
        raise ValueError(f"No keys starting with '{backbone_key}' found in checkpoint.")

    print(f"✅ Extracted {len(backbone_state_dict)} backbone parameters.")

    # Optionally save cleaned state_dict
    if save_path:
        torch.save(backbone_state_dict, save_path)
        print(f"Saved backbone weights to {save_path}")

    return backbone_state_dict


def get_best_checkpoint_by_val_loss(ckpt_dir):
    """
    Returns the filename of the checkpoint with the lowest val_loss.
    Ignores files without 'val_loss=' in the name.
    """
    best_ckpt = None
    best_loss = float("inf")

    pattern = re.compile(r"val_loss=([0-9]*\.?[0-9]+)")

    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".ckpt"):
            continue

        match = pattern.search(fname)
        if match is None:
            continue

        val_loss = float(match.group(1))
        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt = fname

    return best_ckpt


def get_best_checkpoint_by_val_knn_acc(ckpt_dir):
    """
    Returns the filename of the checkpoint with the highest val_knn_acc_epoch.
    Ignores files without 'val_knn_acc_epoch=' in the name.
    """
    best_ckpt = None
    best_acc = float("-inf")

    pattern = re.compile(r"best_knn_epoch=([0-9]*\.?[0-9]+)")

    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".ckpt"):
            continue

        match = pattern.search(fname)
        if match is None:
            continue

        acc = float(match.group(1))
        if acc > best_acc:
            best_acc = acc
            best_ckpt = fname

    return best_ckpt


def get_best_checkpoint_by_mrr(ckpt_dir):
    """
    Returns the filename of the checkpoint with the highest val_knn_acc_epoch.
    Ignores files without 'val_knn_acc_epoch=' in the name.
    """
    best_ckpt = None
    best_acc = float("-inf")

    pattern = re.compile(r"best_mrr_epoch=([0-9]*\.?[0-9]+)")

    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".ckpt"):
            continue

        match = pattern.search(fname)
        if match is None:
            continue

        acc = float(match.group(1))
        if acc > best_acc:
            best_acc = acc
            best_ckpt = fname

    return best_ckpt