import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from byol_pytorch.byol_pytorch import RandomApply, NetWrapper, EMA, MLP, default, get_module_device, update_moving_average, set_requires_grad, singleton


def loss_fn(x, y):
    """
    Computes the negative cosine similarity loss between two sets of vectors.

    Args:
        x (Tensor): Predicted feature vectors of shape (batch_size, dim).
        y (Tensor): Target feature vectors of shape (batch_size, dim).

    Returns:
        Tensor: A loss tensor of shape (batch_size,) with values in [0, 4], where
                lower values indicate more similar (aligned) vectors.
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def variance_loss(z, eps=1e-4):
    """
    z: (batch_size, dim)
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z):
    """
    z: (batch_size, dim)
    """
    z = z - z.mean(dim=0)
    batch_size, dim = z.shape

    cov = (z.T @ z) / (batch_size - 1)
    off_diag = cov.flatten()[1:].view(dim - 1, dim + 1)[:, :-1]

    return (off_diag ** 2).sum() / dim


class BYOLWithTwoImages(nn.Module):
    """
    https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
    Modified to take two separate image inputs instead of applying augmentations internally
    """
    def __init__(
        self,
        net,
        image_size,
        image_channels = 3,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        sync_batchnorm = None,
        use_vitreg = False, 
        lambda_var_emb = 10.0,
        lambda_cov_emb = 0.5,
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer = hidden_layer,
            use_simsiam_mlp = not use_momentum,
            sync_batchnorm = sync_batchnorm
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, image_channels, image_size, image_size, device=device))

        # VITReg
        self.use_vitreg = use_vitreg
        self.lambda_var_emb = lambda_var_emb
        self.lambda_cov_emb = lambda_cov_emb


    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x1,
        x2=None,
        return_embedding=False,
        return_projection=True
    ):
        
        assert not (self.training and x1.shape[0] == 1), \
            'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
               
        if x2 is None: # Single samples
            if return_embedding:
                return self.online_encoder(x1, return_projection = return_projection)

            # Image augmentation to create two distinct samples from x1
            image_one, image_two = self.augment1(x1), self.augment2(x1)
            images = torch.cat((image_one, image_two), dim = 0)

        else: # Sample pairs
            if return_embedding:
                embedding1 = self.online_encoder(x1, return_projection=return_projection)
                embedding2 = self.online_encoder(x2, return_projection=return_projection)
                return embedding1, embedding2

            # No augmentation here
            images = torch.cat((x1, x2), dim=0)

        online_projections, online_embeddings = self.online_encoder(images) # Backbone + Projector
        online_predictions = self.online_predictor(online_projections) # Predictor head

        online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0) # Split samples

        # Target encoder is either the momentum updated net or just the online encoder
        with torch.no_grad(): 
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_projections, _ = target_encoder(images) # Backbone + Projector
            target_projections = target_projections.detach()

            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0) # Split

        loss_one = loss_fn(online_pred_one, target_proj_two.detach()) # Loss online1 to target2
        loss_two = loss_fn(online_pred_two, target_proj_one.detach()) # Loss online2 to target1

        loss = (loss_one + loss_two).mean()

        # VICReg on embeddings
        if self.use_vitreg:
            emb_one, emb_two = online_embeddings.chunk(2, dim=0)
            var_emb = variance_loss(emb_one) + variance_loss(emb_two)
            cov_emb = covariance_loss(emb_one) + covariance_loss(emb_two)

            loss = (
                loss
                + var_emb * self.lambda_var_emb
                + cov_emb * self.lambda_cov_emb
                )
        
        return loss