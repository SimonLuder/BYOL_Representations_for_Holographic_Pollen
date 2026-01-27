import torch
import torch.nn as nn
from byol_pytorch.byol_pytorch import NetWrapper
from . import backbones


class BYOLPolenoEmbedding(nn.Module):
    """
    Encoder that uses the backbone encoder pretrained with BYOL
    """
    def __init__(self, ckpt_path: str, emb_layer, out_dim, backbone="resnet50", projection_size=256, projection_hidden_size=4096):
        super().__init__()

        # 1. Build backbone architecture
        net = backbones.get_backbone(backbone, pretrained=False)
        net = backbones.set_single_channel_input(net)
        net = backbones.update_linear_layer(net, layer=emb_layer, out_features=out_dim)

        # 2. Load BYOL-trained weights
        net = backbones.load_byol_backbone_weights_from_checkpoint(ckpt_path, net)

        # 3. Wrap backbone
        self.encoder = NetWrapper(net, layer=emb_layer, projection_size=projection_size, projection_hidden_size=projection_hidden_size)  
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x, return_projection=False)
