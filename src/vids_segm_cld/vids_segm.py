"""
VIDS-Seg: Variational Inference under Distribution Shifts — Segmentation
PyTorch Implementation

Extended from VIDS (Slavutsky & Blei, NeurIPS 2025) to dense pixel-wise prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict
import math


# =============================================================================
# 1. U-Net Building Blocks (from your unet_parts)
# =============================================================================

class DoubleConv(nn.Module):
    """(Conv2d => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# =============================================================================
# 2. Embedding Networks
# =============================================================================

class FCEmbedding(nn.Module):
    """Fully connected embedding network for tabular data."""
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvEmbedding(nn.Module):
    """Convolutional embedding network for image data (non-segmentation)."""
    def __init__(self, in_channels: int = 3, embedding_dim: int = 16):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = None
        self.embedding_dim = embedding_dim

    def _init_fc(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.conv_blocks(x)
            flat_dim = out.view(out.size(0), -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim),
        ).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc is None:
            self._init_fc(x)
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class UNetDenseEmbedding(nn.Module):
    """
    U-Net encoder-decoder that produces dense per-pixel embeddings.
    
    Output shape: (B, embedding_dim, H, W)
    
    This replaces the final classification head of a standard U-Net with
    a feature output layer. The per-pixel embeddings are what θ (the 
    Bayesian prediction head) operates on.
    """
    def __init__(self, n_channels: int = 1, embedding_dim: int = 32, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # Final layer outputs embedding_dim channels instead of n_classes
        self.out_conv = nn.Conv2d(64, embedding_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            embeddings: (B, embedding_dim, H, W) dense per-pixel embeddings
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        embeddings = self.out_conv(x)
        return embeddings


# =============================================================================
# 3. Prediction Heads
# =============================================================================

class PredictionHead(nn.Module):
    """
    Linear prediction head for non-segmentation tasks.
    f_θ(g(x)) = θ^T g(x) + bias
    """
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_params = embedding_dim * output_dim + output_dim

    def forward(self, embeddings: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embedding_dim)
            theta: (num_params,) or (num_samples, num_params)
        Returns:
            predictions: (batch, output_dim) or (num_samples, batch, output_dim)
        """
        w_size = self.embedding_dim * self.output_dim

        if theta.dim() == 1:
            W = theta[:w_size].view(self.embedding_dim, self.output_dim)
            b = theta[w_size:]
            return embeddings @ W + b
        else:
            S = theta.size(0)
            W = theta[:, :w_size].view(S, self.embedding_dim, self.output_dim)
            b = theta[:, w_size:].unsqueeze(1)
            emb = embeddings.unsqueeze(0)
            return torch.bmm(emb.expand(S, -1, -1), W) + b


class SegmentationPredictionHead(nn.Module):
    """
    Per-pixel linear prediction head for segmentation.
    
    Equivalent to a 1×1 convolution: for each pixel, applies the same
    linear transformation θ to the embedding_dim-dimensional feature vector
    to produce num_classes logits.
    
    θ consists of:
        - Weight: (embedding_dim, num_classes)  
        - Bias: (num_classes,)
    Total params = embedding_dim * num_classes + num_classes
    
    This is the layer whose weights are treated as random variables
    in the Bayesian framework. The same θ is shared across all pixels,
    which is what makes this tractable — θ is small even though the
    output is spatially dense.
    """
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_params = embedding_dim * num_classes + num_classes

    def forward(
        self, 
        embeddings: torch.Tensor, 
        theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, embedding_dim, H, W) dense per-pixel embeddings
            theta: (num_params,) single parameter vector
                   or (S, num_params) multiple samples
        
        Returns:
            If theta is 1D: (B, num_classes, H, W) logits
            If theta is 2D: (S, B, num_classes, H, W) logits
        """
        w_size = self.embedding_dim * self.num_classes

        if theta.dim() == 1:
            # Single theta — equivalent to 1×1 conv
            W = theta[:w_size].view(self.num_classes, self.embedding_dim, 1, 1)
            b = theta[w_size:].view(self.num_classes)
            # F.conv2d: weight shape (out_channels, in_channels, kH, kW)
            return F.conv2d(embeddings, W, b)
        
        elif theta.dim() == 2:
            # Multiple theta samples: (S, num_params)
            S = theta.size(0)
            B, D, H, W_spatial = embeddings.shape

            W = theta[:, :w_size].view(S, self.num_classes, self.embedding_dim)
            b = theta[:, w_size:].view(S, self.num_classes)

            # Reshape embeddings: (B, D, H, W) -> (B, D, H*W) -> (B, H*W, D)
            emb_flat = embeddings.view(B, D, -1).permute(0, 2, 1)  # (B, HW, D)
            
            # For each sample s, compute emb_flat @ W[s].T + b[s]
            # W[s]: (num_classes, D) -> W[s].T: (D, num_classes)
            # Result per sample: (B, HW, num_classes)
            results = []
            for s in range(S):
                logits_flat = emb_flat @ W[s].T + b[s]  # (B, HW, num_classes)
                logits = logits_flat.permute(0, 2, 1).view(B, self.num_classes, H, W_spatial)
                results.append(logits)
            
            return torch.stack(results, dim=0)  # (S, B, num_classes, H, W)


# =============================================================================
# 4. Inference Network h_γ
# =============================================================================

class InferenceNetwork(nn.Module):
    """
    Amortized inference network h_γ.
    Takes concatenated [g_bar(x_{1:n}), g(x*)] and outputs
    variational parameters φ = (μ, log_σ) of q_φ(θ | x_{1:n}, x*).
    
    For segmentation, both train_summary and test_summary are global
    vectors obtained by spatially averaging the dense embeddings.
    """
    def __init__(
        self,
        embedding_dim: int,
        theta_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        input_dim = 2 * embedding_dim

        if hidden_dims is None:
            d = theta_dim
            hidden_dims = [64 * d, 32 * d, 16 * d, 8 * d, 4 * d]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, theta_dim)
        self.log_sigma_head = nn.Linear(prev_dim, theta_dim)

    def forward(
        self, train_summary: torch.Tensor, test_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            train_summary: (embedding_dim,) aggregated training embedding
            test_embedding: (M, embedding_dim) or (embedding_dim,) test embeddings

        Returns:
            mu: (M, theta_dim) or (theta_dim,)
            log_sigma: (M, theta_dim) or (theta_dim,)
        """
        if test_embedding.dim() == 1:
            test_embedding = test_embedding.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        M = test_embedding.size(0)
        if train_summary.dim() == 1:
            train_summary = train_summary.unsqueeze(0)
        train_summary_expanded = train_summary.expand(M, -1)

        combined = torch.cat([train_summary_expanded, test_embedding], dim=-1)
        h = self.shared(combined)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)

        if squeeze:
            mu = mu.squeeze(0)
            log_sigma = log_sigma.squeeze(0)

        return mu, log_sigma


# =============================================================================
# 5. Adaptive Prior (Energy-based prior)
# =============================================================================

class AdaptivePrior(nn.Module):
    """
    Energy-based adaptive prior p(θ | x_{1:N}, x*).
    
    For segmentation (pixel-wise classification):
    E(θ; x_{1:N}, x*) = Σ_i Σ_pixel Σ_class log p(y_pixel=class | x_i, θ)
                       + Σ_pixel Σ_class log p(y_pixel=class | x*, θ)
    """
    def __init__(
        self,
        task: str = "classification",
        num_classes: int = 2,
        mc_samples: int = 50,
        y_min: float = -5.0,
        y_max: float = 5.0,
    ):
        super().__init__()
        assert task in ["classification", "regression", "segmentation"]
        self.task = task
        self.num_classes = num_classes
        self.mc_samples = mc_samples
        self.y_min = y_min
        self.y_max = y_max

    def compute_energy(
        self,
        train_logits: torch.Tensor,
        test_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.task == "classification":
            return self._energy_classification(train_logits, test_logits)
        elif self.task == "segmentation":
            return self._energy_segmentation(train_logits, test_logits)
        else:
            return self._energy_regression(train_logits, test_logits)

    def _energy_classification(
        self, train_logits: torch.Tensor, test_logits: torch.Tensor
    ) -> torch.Tensor:
        train_log_probs = F.log_softmax(train_logits, dim=-1)
        train_energy = train_log_probs.sum()

        if test_logits.dim() == 1:
            test_logits = test_logits.unsqueeze(0)
        test_log_probs = F.log_softmax(test_logits, dim=-1)
        test_energy = test_log_probs.sum()

        return train_energy + test_energy

    def _energy_segmentation(
        self, train_logits: torch.Tensor, test_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Segmentation energy: sum of per-pixel classification energies.
        
        Args:
            train_logits: (N, C, H, W) or (B, N, C, H, W)
            test_logits: (B, C, H, W) or (C, H, W)
        """
        # log softmax over classes dimension (dim=1 for NCHW)
        if train_logits.dim() == 4:
            train_log_probs = F.log_softmax(train_logits, dim=1)  # (N, C, H, W)
        else:
            train_log_probs = F.log_softmax(train_logits, dim=-3)
        train_energy = train_log_probs.sum()

        if test_logits.dim() == 3:
            test_logits = test_logits.unsqueeze(0)
        test_log_probs = F.log_softmax(test_logits, dim=1)
        test_energy = test_log_probs.sum()

        return train_energy + test_energy

    def _energy_regression(
        self, train_preds: torch.Tensor, test_preds: torch.Tensor
    ) -> torch.Tensor:
        y_samples = torch.linspace(
            self.y_min, self.y_max, self.mc_samples, device=train_preds.device
        )
        train_mu = train_preds.squeeze(-1)
        diff_train = y_samples.unsqueeze(1) - train_mu.unsqueeze(0)
        log_lik_train = -0.5 * diff_train**2 - 0.5 * math.log(2 * math.pi)
        scale = (self.y_max - self.y_min) / self.mc_samples
        train_energy = (log_lik_train.sum(dim=1) * scale).sum()

        if test_preds.dim() == 1:
            test_preds = test_preds.unsqueeze(0)
        test_mu = test_preds.squeeze(-1)
        diff_test = y_samples.unsqueeze(1) - test_mu.unsqueeze(0)
        log_lik_test = -0.5 * diff_test**2 - 0.5 * math.log(2 * math.pi)
        test_energy = (log_lik_test.sum(dim=1) * scale).sum()

        return train_energy + test_energy

    def log_prior(
        self,
        train_logits: torch.Tensor,
        test_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_energy(train_logits, test_logits)


# =============================================================================
# 6. ELBO Computation
# =============================================================================

class ELBOComputer(nn.Module):
    """
    Computes the ELBO objective.
    
    For segmentation, the log-likelihood is the sum of per-pixel
    cross-entropy losses across all spatial locations.
    """
    def __init__(
        self,
        task: str = "classification",
        kl_weight: float = 1.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.task = task
        self.kl_weight = kl_weight
        self.num_classes = num_classes
        self.prior = AdaptivePrior(
            task=task,
            num_classes=num_classes,
        )

    def forward(
        self,
        train_x_emb: torch.Tensor,
        train_y: torch.Tensor,
        test_x_emb: torch.Tensor,
        theta: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        prediction_head,
    ) -> torch.Tensor:
        """
        Args:
            For non-segmentation:
                train_x_emb: (N, d) embedded training covariates
                train_y: (N,) training labels
                test_x_emb: (d,) embedded test covariate
            For segmentation:
                train_x_emb: (N, d, H, W) dense training embeddings
                train_y: (N, H, W) training segmentation masks
                test_x_emb: (1, d, H, W) dense test embeddings
            theta: (theta_dim,) sampled parameters
            mu: (theta_dim,) variational mean
            log_sigma: (theta_dim,) variational log std
            prediction_head: the prediction head module

        Returns:
            elbo: scalar
        """
        if self.task == "segmentation":
            return self._forward_segmentation(
                train_x_emb, train_y, test_x_emb,
                theta, mu, log_sigma, prediction_head
            )
        else:
            return self._forward_standard(
                train_x_emb, train_y, test_x_emb,
                theta, mu, log_sigma, prediction_head
            )

    def _forward_standard(
        self, train_x_emb, train_y, test_x_emb,
        theta, mu, log_sigma, prediction_head
    ):
        # 1. Log-likelihood
        train_preds = prediction_head(train_x_emb, theta)
        log_lik = self._log_likelihood(train_preds, train_y)

        # 2. Log prior
        test_preds = prediction_head(
            test_x_emb.unsqueeze(0) if test_x_emb.dim() == 1 else test_x_emb,
            theta,
        )
        log_prior = self.prior.log_prior(train_preds, test_preds)

        # 3. Log q
        sigma = torch.exp(log_sigma)
        log_q_sample = torch.sum(
            -0.5 * ((theta - mu) / sigma) ** 2
            - log_sigma
            - 0.5 * math.log(2 * math.pi)
        )

        elbo = log_lik + self.kl_weight * (log_prior - log_q_sample)
        return elbo

    def _forward_segmentation(
        self, train_x_emb, train_y, test_x_emb,
        theta, mu, log_sigma, prediction_head
    ):
        """
        Segmentation-specific ELBO.
        
        Args:
            train_x_emb: (N, d, H, W)
            train_y: (N, H, W) with integer class labels
            test_x_emb: (1, d, H, W)
            theta: (theta_dim,)
        """
        # 1. Log-likelihood: per-pixel cross-entropy summed over all pixels and images
        train_logits = prediction_head(train_x_emb, theta)  # (N, C, H, W)
        log_lik = self._log_likelihood_segmentation(train_logits, train_y)

        # 2. Log prior (energy-based)
        test_logits = prediction_head(test_x_emb, theta)  # (1, C, H, W)
        log_prior = self.prior.log_prior(train_logits, test_logits)

        # 3. Log q
        sigma = torch.exp(log_sigma)
        log_q_sample = torch.sum(
            -0.5 * ((theta - mu) / sigma) ** 2
            - log_sigma
            - 0.5 * math.log(2 * math.pi)
        )

        elbo = log_lik + self.kl_weight * (log_prior - log_q_sample)
        return elbo

    def _log_likelihood(self, preds, targets):
        if self.task == "classification":
            return -F.cross_entropy(preds, targets.long(), reduction="sum")
        else:
            return -0.5 * torch.sum((preds - targets) ** 2)

    def _log_likelihood_segmentation(self, logits, targets):
        """
        Per-pixel cross-entropy, summed over all pixels and images.
        
        Args:
            logits: (N, C, H, W)
            targets: (N, H, W) long tensor
        """
        return -F.cross_entropy(logits, targets.long(), reduction="sum")


# =============================================================================
# 7. Synthetic Environment Generator
# =============================================================================

class SyntheticEnvironmentGenerator:
    """
    Generates synthetic environments by bootstrap subsampling.
    Works for both tabular and image segmentation data.
    """
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        n_train: int,
        n_test: int,
    ):
        self.train_x = train_x
        self.train_y = train_y
        self.N = train_x.size(0)
        self.n_train = n_train
        self.n_test = n_test

    def sample_environment(self):
        train_idx = torch.randint(0, self.N, (self.n_train,))
        test_idx = torch.randint(0, self.N, (self.n_test,))
        return (
            self.train_x[train_idx],
            self.train_y[train_idx],
            self.train_x[test_idx],
            self.train_y[test_idx],
        )


# =============================================================================
# 8. VIDS Model (Main class — supports all tasks)
# =============================================================================

class VIDS(nn.Module):
    """
    Variational Inference under Distribution Shifts (VIDS).
    
    Supports:
    - 'classification': standard image/tabular classification
    - 'regression': tabular regression
    - 'segmentation': dense pixel-wise classification (U-Net based)
    """

    def __init__(
        self,
        embedding_net: nn.Module,
        embedding_dim: int,
        output_dim: int,
        task: str = "classification",
        inference_hidden_dims: Optional[List[int]] = None,
        kl_weight: float = 0.005,
        variance_penalty: float = 0.001,
        num_classes: int = 2,
    ):
        """
        Args:
            embedding_net: pre-trained embedding network g_ξ
                For segmentation: should be UNetDenseEmbedding producing (B, embedding_dim, H, W)
                For others: produces (B, embedding_dim)
            embedding_dim: dimension of embeddings
            output_dim: number of output classes (classification/segmentation) or 1 (regression)
            task: 'classification', 'regression', or 'segmentation'
            inference_hidden_dims: hidden layer sizes for h_γ
            kl_weight: weight λ for KL term
            variance_penalty: τ for cross-environment variance penalty
            num_classes: number of classes
        """
        super().__init__()

        self.embedding_net = embedding_net
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.task = task
        self.variance_penalty = variance_penalty

        # Choose appropriate prediction head
        if task == "segmentation":
            self.prediction_head = SegmentationPredictionHead(embedding_dim, output_dim)
        else:
            self.prediction_head = PredictionHead(embedding_dim, output_dim)
        
        theta_dim = self.prediction_head.num_params

        # Inference network
        self.inference_net = InferenceNetwork(
            embedding_dim=embedding_dim,
            theta_dim=theta_dim,
            hidden_dims=inference_hidden_dims,
        )

        # ELBO computer
        self.elbo_computer = ELBOComputer(
            task=task,
            kl_weight=kl_weight,
            num_classes=num_classes,
        )

        # Freeze embedding network
        for param in self.embedding_net.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def compute_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings using the frozen pre-trained network.
        
        For segmentation: returns (B, embedding_dim, H, W)
        For others: returns (B, embedding_dim)
        """
        self.embedding_net.eval()
        return self.embedding_net(x)

    def aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate embeddings into a single summary vector.
        
        For segmentation (B, D, H, W): global average pool over spatial dims,
            then average over batch -> (D,)
        For tabular/image classification (B, D): average over batch -> (D,)
        """
        if self.task == "segmentation":
            # Global average pool: (B, D, H, W) -> (B, D)
            pooled = embeddings.mean(dim=(2, 3))
            # Average over batch: (B, D) -> (D,)
            return pooled.mean(dim=0)
        else:
            return embeddings.mean(dim=0)

    def aggregate_single_image_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        For segmentation: spatially pool a single image's dense embedding 
        to get a global vector for the inference network.
        
        (D, H, W) -> (D,)  or  (1, D, H, W) -> (D,)
        """
        if embedding.dim() == 4:
            return embedding.mean(dim=(0, 2, 3))
        elif embedding.dim() == 3:
            return embedding.mean(dim=(1, 2))
        else:
            return embedding

    def sample_theta(self, mu, log_sigma):
        """Reparameterization trick."""
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def compute_environment_elbo(
        self,
        env_train_x: torch.Tensor,
        env_train_y: torch.Tensor,
        env_test_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ELBO for a single synthetic environment.
        """
        # Compute embeddings
        train_emb = self.compute_embeddings(env_train_x)
        test_emb = self.compute_embeddings(env_test_x)

        # Aggregate training embeddings into summary vector
        train_summary = self.aggregate_embeddings(train_emb)

        total_elbo = torch.tensor(0.0, device=env_train_x.device)

        if self.task == "segmentation":
            # For segmentation, iterate over test images
            for j in range(env_test_x.size(0)):
                test_emb_j = test_emb[j:j+1]  # (1, D, H, W) — keep batch dim
                
                # Pool to get global vector for inference network
                test_summary_j = self.aggregate_single_image_embedding(test_emb_j)
                
                # Get variational parameters
                mu, log_sigma = self.inference_net(train_summary, test_summary_j)
                
                # Sample θ
                theta = self.sample_theta(mu, log_sigma)
                
                # Compute ELBO
                elbo_j = self.elbo_computer(
                    train_x_emb=train_emb,
                    train_y=env_train_y if env_train_y.dim() > 1 else env_train_y,
                    test_x_emb=test_emb_j,
                    theta=theta,
                    mu=mu,
                    log_sigma=log_sigma,
                    prediction_head=self.prediction_head,
                )
                total_elbo = total_elbo + elbo_j
        else:
            # Original non-segmentation path
            for j in range(env_test_x.size(0)):
                test_emb_j = test_emb[j]
                mu, log_sigma = self.inference_net(train_summary, test_emb_j)
                theta = self.sample_theta(mu, log_sigma)
                elbo_j = self.elbo_computer(
                    train_x_emb=train_emb,
                    train_y=env_train_y,
                    test_x_emb=test_emb_j,
                    theta=theta,
                    mu=mu,
                    log_sigma=log_sigma,
                    prediction_head=self.prediction_head,
                )
                total_elbo = total_elbo + elbo_j

        return total_elbo

    def compute_loss(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        num_environments: int,
        env_train_size: int,
        env_test_size: int,
    ) -> torch.Tensor:
        """
        Compute full VIDS training loss with cross-environment variance penalty.
        """
        env_generator = SyntheticEnvironmentGenerator(
            train_x, train_y, env_train_size, env_test_size
        )

        env_elbos = []
        for _ in range(num_environments):
            env_train_x, env_train_y, env_test_x, env_test_y = (
                env_generator.sample_environment()
            )
            elbo = self.compute_environment_elbo(
                env_train_x, env_train_y, env_test_x
            )
            env_elbos.append(elbo)

        env_elbos = torch.stack(env_elbos)
        mean_elbo = env_elbos.mean()
        var_penalty = env_elbos.var() if num_environments > 1 else torch.tensor(0.0)
        loss = -mean_elbo + self.variance_penalty * var_penalty
        return loss

    @torch.no_grad()
    def predict(
        self,
        train_x: torch.Tensor,
        test_x: torch.Tensor,
        num_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation.
        
        For segmentation:
            Returns per-pixel predictions and uncertainty maps.
        """
        self.eval()

        train_emb = self.compute_embeddings(train_x)
        test_emb = self.compute_embeddings(test_x)
        train_summary = self.aggregate_embeddings(train_emb)

        if self.task == "segmentation":
            return self._predict_segmentation(
                train_summary, test_emb, num_samples
            )
        else:
            return self._predict_standard(
                train_summary, test_emb, test_x, num_samples
            )

    def _predict_standard(self, train_summary, test_emb, test_x, num_samples):
        M = test_x.size(0)
        all_predictions = []

        for j in range(M):
            test_emb_j = test_emb[j]
            mu, log_sigma = self.inference_net(train_summary, test_emb_j)
            sigma = torch.exp(log_sigma)
            eps = torch.randn(num_samples, mu.size(0), device=mu.device)
            theta_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
            preds = self.prediction_head(
                test_emb_j.unsqueeze(0), theta_samples
            )
            all_predictions.append(preds.squeeze(1))

        all_predictions = torch.stack(all_predictions, dim=1)

        if self.task == "classification":
            probs = F.softmax(all_predictions, dim=-1)
            mean_probs = probs.mean(dim=0)
            std_probs = probs.std(dim=0)
            return {
                "predictions": mean_probs,
                "std": std_probs,
                "samples": probs,
                "predicted_classes": mean_probs.argmax(dim=-1),
            }
        else:
            mean_preds = all_predictions.mean(dim=0)
            std_preds = all_predictions.std(dim=0)
            return {
                "predictions": mean_preds,
                "std": std_preds,
                "samples": all_predictions,
            }

    def _predict_segmentation(self, train_summary, test_emb, num_samples):
        """
        Segmentation prediction with per-pixel uncertainty.
        
        Args:
            train_summary: (D,) aggregated training summary
            test_emb: (M, D, H, W) test image embeddings
            num_samples: number of posterior samples
        
        Returns:
            dict with:
                'predictions': (M, C, H, W) mean class probabilities
                'std': (M, C, H, W) standard deviation of probabilities
                'predicted_masks': (M, H, W) argmax predictions
                'uncertainty_map': (M, H, W) predictive entropy
                'samples': (S, M, C, H, W) all probability samples
        """
        M = test_emb.size(0)
        all_probs = []  # Will collect (S, 1, C, H, W) for each test image

        for j in range(M):
            test_emb_j = test_emb[j:j+1]  # (1, D, H, W)
            test_summary_j = self.aggregate_single_image_embedding(test_emb_j)

            mu, log_sigma = self.inference_net(train_summary, test_summary_j)
            sigma = torch.exp(log_sigma)
            eps = torch.randn(num_samples, mu.size(0), device=mu.device)
            theta_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, theta_dim)

            # Compute logits for all samples
            logits = self.prediction_head(test_emb_j, theta_samples)  # (S, 1, C, H, W)
            probs = F.softmax(logits, dim=2)  # softmax over classes
            all_probs.append(probs.squeeze(1))  # (S, C, H, W)

        # Stack over test images: (S, M, C, H, W)
        all_probs = torch.stack(all_probs, dim=1)

        mean_probs = all_probs.mean(dim=0)  # (M, C, H, W)
        std_probs = all_probs.std(dim=0)    # (M, C, H, W)
        predicted_masks = mean_probs.argmax(dim=1)  # (M, H, W)

        # Predictive entropy: H[p(y|x)] = -Σ_c p_c log p_c
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)  # (M, H, W)

        # Mutual information (epistemic uncertainty):
        # MI = H[E_q[p(y|x,θ)]] - E_q[H[p(y|x,θ)]]
        per_sample_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2)  # (S, M, H, W)
        expected_entropy = per_sample_entropy.mean(dim=0)  # (M, H, W) — aleatoric
        mutual_info = entropy - expected_entropy  # (M, H, W) — epistemic

        return {
            "predictions": mean_probs,
            "std": std_probs,
            "predicted_masks": predicted_masks,
            "uncertainty_map": entropy,
            "aleatoric_uncertainty": expected_entropy,
            "epistemic_uncertainty": mutual_info,
            "samples": all_probs,
        }


# =============================================================================
# 9. Pre-training utilities
# =============================================================================

def pretrain_embedding(
    embedding_net, prediction_head_pretrain,
    train_x, train_y,
    task="regression", epochs=100, lr=1e-3, batch_size=64,
):
    """
    Pre-train the embedding network.
    
    For segmentation:
        embedding_net: UNetDenseEmbedding
        prediction_head_pretrain: nn.Conv2d(embedding_dim, num_classes, 1)
        train_x: (N, C, H, W)
        train_y: (N, H, W) integer masks
    """
    device = train_x.device

    model = nn.Sequential(embedding_net, prediction_head_pretrain).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            if task == "classification":
                loss = F.cross_entropy(preds, y_batch.long())
            elif task == "segmentation":
                loss = F.cross_entropy(preds, y_batch.long())
            else:
                loss = F.mse_loss(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % max(1, epochs // 10) == 0:
            avg_loss = total_loss / n_batches
            print(f"  Pre-train Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        if task == "segmentation":
            # Compute mean dice or IoU on a small batch
            sample_x = train_x[:min(8, len(train_x))]
            sample_y = train_y[:min(8, len(train_y))]
            sample_preds = model(sample_x)
            sample_loss = F.cross_entropy(sample_preds, sample_y.long())
            pred_masks = sample_preds.argmax(dim=1)
            accuracy = (pred_masks == sample_y.long()).float().mean()
            print(f"  Final train CE: {sample_loss.item():.6f}, pixel acc: {accuracy.item():.4f}")
        else:
            final_preds = model(train_x)
            if task == "classification":
                final_loss = F.cross_entropy(final_preds, train_y.long())
                print(f"  Final train CE: {final_loss.item():.6f}")
            else:
                final_mse = F.mse_loss(final_preds, train_y)
                print(f"  Final train MSE: {final_mse.item():.6f}")

    embedding_net.eval()
    for param in embedding_net.parameters():
        param.requires_grad = False

    return embedding_net, model


def pretrain_segmentation_embedding(
    embedding_net: UNetDenseEmbedding,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int = 2,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 8,
    class_weights: Optional[torch.Tensor] = None,
    dice_alpha: float = 0.5,
):
    """
    Convenience function for pre-training the segmentation embedding 
    with combined CE + Dice loss (matching your original UNet training).
    
    Args:
        embedding_net: UNetDenseEmbedding to pre-train
        train_x: (N, C, H, W) training images
        train_y: (N, H, W) segmentation masks
        num_classes: number of segmentation classes
        epochs: training epochs
        lr: learning rate
        batch_size: batch size
        class_weights: optional class weights for CE loss
        dice_alpha: weight for CE vs Dice (alpha * CE + (1-alpha) * Dice)
    
    Returns:
        embedding_net: frozen pre-trained embedding network
        full_model: the full model (embedding + head) for reference
    """
    device = train_x.device
    
    # 1×1 conv head for pre-training
    pretrain_head = nn.Conv2d(
        embedding_net.embedding_dim, num_classes, kernel_size=1
    ).to(device)
    
    model = nn.Sequential(embedding_net, pretrain_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    smooth = 1e-6
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0
        
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)  # (B, C, H, W)
            
            # CE loss
            ce_loss = F.cross_entropy(
                logits, y_batch.long(), 
                weight=class_weights, 
                reduction="mean"
            )
            
            # Dice loss
            pred_soft = torch.softmax(logits, dim=1)
            target_one_hot = F.one_hot(
                y_batch.long(), num_classes=num_classes
            ).permute(0, 3, 1, 2).float()
            
            intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
            union = torch.sum(pred_soft + target_one_hot, dim=(2, 3))
            dice_score = (2 * intersection + smooth) / (union + smooth)
            dice_loss = 1 - dice_score.mean()
            
            loss = dice_alpha * ce_loss + (1 - dice_alpha) * dice_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_dice += dice_score.mean().item()
            n_batches += 1
        
        if (epoch + 1) % max(1, epochs // 10) == 0:
            avg_loss = total_loss / n_batches
            avg_dice = total_dice / n_batches
            print(
                f"  Pre-train Epoch {epoch+1}/{epochs}, "
                f"Loss: {avg_loss:.6f}, Dice: {avg_dice:.4f}"
            )
    
    # Freeze
    embedding_net.eval()
    for param in embedding_net.parameters():
        param.requires_grad = False
    
    return embedding_net, model


# =============================================================================
# 10. Training Loop
# =============================================================================

def train_vids(
    model: VIDS,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_environments: int = 10,
    env_train_size: int = 500,
    env_test_size: int = 20,
    num_steps: int = 30,
    lr: float = 1e-3,
    verbose: bool = True,
) -> List[float]:
    """
    Train the VIDS model (Algorithm 2 from the paper).
    Only the inference network parameters are optimized.
    """
    optimizer = torch.optim.Adam(model.inference_net.parameters(), lr=lr)

    losses = []
    model.train()

    for step in range(num_steps):
        optimizer.zero_grad()

        loss = model.compute_loss(
            train_x=train_x,
            train_y=train_y,
            num_environments=num_environments,
            env_train_size=min(env_train_size, train_x.size(0)),
            env_test_size=env_test_size,
        )

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (step + 1) % max(1, num_steps // 10) == 0:
            print(f"  Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")

    return losses


# =============================================================================
# 11. Usage Example
# =============================================================================

def example_segmentation():
    """
    Example showing how to use VIDS for image segmentation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Hyperparameters ---
    n_channels = 1       # grayscale
    num_classes = 2       # binary segmentation
    embedding_dim = 32    # per-pixel embedding dimension
    H, W = 128, 128      # image size
    N_train = 50          # number of training images
    
    # --- Create dummy data ---
    train_x = torch.randn(N_train, n_channels, H, W, device=device)
    train_y = torch.randint(0, num_classes, (N_train, H, W), device=device)
    
    # --- Step 1: Create and pre-train embedding network ---
    print("Step 1: Pre-training embedding network...")
    embedding_net = UNetDenseEmbedding(
        n_channels=n_channels, 
        embedding_dim=embedding_dim, 
        bilinear=False
    ).to(device)
    
    embedding_net, pretrained_model = pretrain_segmentation_embedding(
        embedding_net=embedding_net,
        train_x=train_x,
        train_y=train_y,
        num_classes=num_classes,
        epochs=20,
        lr=1e-4,
        batch_size=8,
    )
    
    # --- Step 2: Create VIDS model ---
    print("\nStep 2: Creating VIDS model...")
    
    # For segmentation with small theta, we need small inference network
    theta_dim = embedding_dim * num_classes + num_classes  # 32*2 + 2 = 66
    inference_hidden = [256, 128, 64]  # Smaller than default
    
    vids_model = VIDS(
        embedding_net=embedding_net,
        embedding_dim=embedding_dim,
        output_dim=num_classes,
        task="segmentation",
        inference_hidden_dims=inference_hidden,
        kl_weight=0.001,
        variance_penalty=0.001,
        num_classes=num_classes,
    ).to(device)
    
    print(f"  Prediction head params (θ): {vids_model.prediction_head.num_params}")
    print(f"  Inference network params: {sum(p.numel() for p in vids_model.inference_net.parameters())}")
    
    # --- Step 3: Train VIDS ---
    print("\nStep 3: Training VIDS inference network...")
    losses = train_vids(
        model=vids_model,
        train_x=train_x,
        train_y=train_y,
        num_environments=3,
        env_train_size=min(10, N_train),
        env_test_size=2,
        num_steps=20,
        lr=1e-3,
        verbose=True,
    )
    
    # --- Step 4: Predict with uncertainty ---
    print("\nStep 4: Making predictions...")
    test_x = torch.randn(4, n_channels, H, W, device=device)
    
    results = vids_model.predict(
        train_x=train_x,
        test_x=test_x,
        num_samples=50,
    )
    
    print(f"  Predicted masks shape: {results['predicted_masks'].shape}")
    print(f"  Mean probs shape: {results['predictions'].shape}")
    print(f"  Uncertainty map shape: {results['uncertainty_map'].shape}")
    print(f"  Epistemic uncertainty shape: {results['epistemic_uncertainty'].shape}")
    print(f"  Aleatoric uncertainty shape: {results['aleatoric_uncertainty'].shape}")
    
    # Per-image summary
    for i in range(test_x.size(0)):
        total_unc = results['uncertainty_map'][i].mean().item()
        epist = results['epistemic_uncertainty'][i].mean().item()
        aleat = results['aleatoric_uncertainty'][i].mean().item()
        print(f"  Image {i}: total_uncertainty={total_unc:.4f}, "
              f"epistemic={epist:.4f}, aleatoric={aleat:.4f}")
    
    return results


if __name__ == "__main__":
    example_segmentation()