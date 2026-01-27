import torch
import torch.nn.functional as F


def variance_loss(z, eps=1e-4):
    """
    Variance loss implementation from VICReg paper.
    Args:
        z (Tensor): shape (batch_size, dim)
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def invariance_loss(z1, z2, reduction="mean"):
    """
    VICReg invariance loss.
    
    Args:
        z1 (Tensor): Representations from view 1, shape (N, D)
        z2 (Tensor): Representations from view 2, shape (N, D)
        reduction (str): 'mean' or 'sum'
        
    Returns:
        Tensor: scalar loss
    """
    return F.mse_loss(z1, z2, reduction=reduction)


def covariance_loss(z):
    """
    Covariance loss implementation from VICReg paper.
    Args:
        z (Tensor): shape (batch_size, dim)
    """
    z = z - z.mean(dim=0)
    batch_size, dim = z.shape

    cov = (z.T @ z) / (batch_size - 1)
    off_diag = cov.flatten()[1:].view(dim - 1, dim + 1)[:, :-1]

    return (off_diag ** 2).sum() / dim


def neg_cosine(x, y):
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