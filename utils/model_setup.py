import torch
import torch.nn as nn
from typing import Optional

def set_single_channel_input(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
    """
    Update the model to accept single-channel (grayscale) input images.
    
    Args:
        model (nn.Module): The model to modify.
        layer_name (str, optional): Name of the layer to replace. If None, will auto-detect.
        
    Returns:
        torch.nn.Module: Updated model.
    """
    def convert_conv_to_single_channel(conv: nn.Conv2d) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
            if conv.bias is not None:
                new_conv.bias[:] = conv.bias
        return new_conv
    
    def get_parent_module(root: nn.Module, path: str):
        parts = path.split(".")
        for part in parts[:-1]:
            root = getattr(root, part)
        return root, parts[-1]

    if layer_name:
        parent, attr = get_parent_module(model, layer_name)
        setattr(parent, attr, convert_conv_to_single_channel(getattr(parent, attr)))
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                parent, attr = get_parent_module(model, name)
                setattr(parent, attr, convert_conv_to_single_channel(module))
                break

    return model