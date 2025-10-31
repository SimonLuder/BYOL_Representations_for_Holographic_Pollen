import subprocess
import torch
import requests
from pathlib import Path
import os


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