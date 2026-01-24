from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import variance_loss, invariance_loss, covariance_loss, neg_cosine

# Base Objective
class SSLObjective(nn.Module, ABC):
    """
    Abstract base class for self-supervised learning objectives.

    All objectives operate on TWO views per sample.
    """

    @abstractmethod
    def forward(
        self,
        online_proj_1: torch.Tensor,
        online_proj_2: torch.Tensor,
        target_proj_1: Optional[torch.Tensor] = None,
        target_proj_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            online_proj_1, online_proj_2:
                Projections from the online encoder.
            target_proj_1, target_proj_2:
                Projections from the target encoder (if applicable).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError


# BYOL & SimSiam
class NormalizedL2Objective(SSLObjective):
    """
    BYOL and SimSiam objective.
    """

    def __init__(self):
        """
        Args:
            loss_fn:
                A similarity loss (e.g. negative cosine similarity).
        """
        super().__init__()
        self.loss_fn = neg_cosine

    def forward(
        self,
        o1: torch.Tensor,
        o2: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        ) -> torch.Tensor:

    
        if t1 is None or t2 is None:
            raise ValueError("BYOLObjective requires target projections")

        loss_1 = self.loss_fn(o1, t2.detach())
        loss_2 = self.loss_fn(o2, t1.detach())

        return (loss_1 + loss_2).mean()

# VICReg
class VICRegObjective(SSLObjective):
    """
    Variance-Invariance-Covariance Regularization (VICReg).
    """

    def __init__(
        self,
        lambda_inv: float = 1.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(
        self,
        o1: torch.Tensor,
        o2: torch.Tensor,
        *_,
        **__,
    ) -> torch.Tensor:
        inv = invariance_loss(o1, o2)
        var = variance_loss(o1) + variance_loss(o2)
        cov = covariance_loss(o1) + covariance_loss(o2)

        return (
            self.lambda_inv * inv
            + self.lambda_var * var
            + self.lambda_cov * cov
        )
    

class HybridObjective(SSLObjective):
    """
    Hybrid objective:
    - SimSiam or Byol style asymmetric invariance
    - VICReg variance + covariance regularization
    """

    def __init__(
        self,
        lambda_inv: float = 1.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(
        self,
        o1: torch.Tensor,
        o2: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        inv = invariance_loss(o1, t2)
        var = variance_loss(o1) + variance_loss(o2)
        cov = covariance_loss(o1) + covariance_loss(o2)

        return (
            self.lambda_inv * inv
            + self.lambda_var * var
            + self.lambda_cov * cov
        )
    

class Hybrid2Objective(SSLObjective):
    """
    Hybrid objective:
    - SimSiam or Byol style asymmetric invariance
    - VICReg variance + covariance regularization
    """

    def __init__(
        self,
        lambda_inv: float = 1.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(
        self,
        o1: torch.Tensor,
        o2: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        # BYOL-style invariance (stop-grad happens upstream)
        inv_1 = invariance_loss(o1, t2)
        inv_2 = invariance_loss(o2, t1)
        inv = 0.5 * (inv_1 + inv_2)

        # VICReg regularization (online only)
        var = variance_loss(o1) + variance_loss(o2)
        cov = covariance_loss(o1) + covariance_loss(o2)

        return (
            self.lambda_inv * inv
            + self.lambda_var * var
            + self.lambda_cov * cov
        )