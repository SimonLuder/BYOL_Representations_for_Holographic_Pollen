import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from byol_pytorch.byol_pytorch import RandomApply, NetWrapper, EMA, MLP, default, get_module_device, update_moving_average, set_requires_grad, singleton


class SelfSupervisedLearner(nn.Module):
    """
    https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
    Modified to take two separate image inputs instead of applying augmentations internally
    """
    def __init__(
        self,
        net,
        image_size,
        objective=None,
        image_channels = 3,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        use_prediction_head = True,
        sync_batchnorm = None,
        ):
        super().__init__()
        self.net = net
        self.objective = objective
        self.use_momentum = use_momentum
        self.use_prediction_head = use_prediction_head

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

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, 
            projection_size, 
            projection_hidden_size
        )

        # Device
        device = get_module_device(net)
        self.to(device)

        # Trigger lazy modules
        self.forward(torch.randn(2, image_channels, image_size, image_size, device=device))


    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder


    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None


    def update_moving_average(self):
        assert self.use_momentum, 'Momentum encoder disabled — no EMA update needed'
        assert self.target_encoder is not None, 'Target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)


    def forward(
        self,
        x1,
        x2=None,
        return_embedding=False,
        return_projection=True,
        validation=False,
        ):
        
        assert not (self.training and x1.shape[0] == 1), \
            'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
               
        # Image augmentations and preprocessing  
        if x2 is None: # Single image
            if return_embedding and not validation:
                return self.online_encoder(x1, return_projection = return_projection)

            # Image augmentation to create two distinct samples from x1
            v1, v2 = self.augment1(x1), self.augment2(x1)
            images = torch.cat((v1, v2), dim = 0)

            if return_embedding:
                embedding1 = self.online_encoder(v1, return_projection=return_projection)
                embedding2 = self.online_encoder(v2, return_projection=return_projection)
                return embedding1, embedding2

        else: # Paired images
            if return_embedding:
                embedding1 = self.online_encoder(x1, return_projection=return_projection)
                embedding2 = self.online_encoder(x2, return_projection=return_projection)
                return embedding1, embedding2

            # No augmentation here
            images = torch.cat((x1, x2), dim=0)

        # Online forward
        online_proj, _ = self.online_encoder(images) # Backbone + Projector return (projections, embeddings)
        online_proj_1, online_proj_2 = online_proj.chunk(2, dim=0)

        # Prediction head (BYOL & SimSiam)
        if self.use_prediction_head:
            online_pred = self.online_predictor(online_proj) # Predictor head
            online_pred_1, online_pred_2 = online_pred.chunk(2, dim=0)

        # Target forward
        with torch.no_grad(): 
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj, _ = target_encoder(images) # Backbone + Projector
            target_proj = target_proj.detach()
            target_proj_1, target_proj_2 = target_proj.chunk(2, dim=0)

        if self.objective is None:
            return

        # Loss objective
        if self.use_prediction_head:
            loss = self.objective(
                o1=online_pred_1, 
                o2=online_pred_2,
                t1=target_proj_1,
                t2=target_proj_2
            )
        else:
            loss = self.objective(
                o1=online_proj_1, 
                o2=online_proj_2,
                t1=target_proj_1,
                t2=target_proj_2
            )

        return loss