"""
VIDS-Seg: Variational Inference under Distribution Shifts — Segmentation
PyTorch Lightning Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
import math
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List, Dict
from unet.unet_parts import *


# =============================================================================
# 2. Embedding Networks (unchanged — these are used outside Lightning)
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
        self.out_conv = nn.Conv2d(64, embedding_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
# 3. Prediction Heads (unchanged)
# =============================================================================

class PredictionHead(nn.Module):
    """Linear prediction head for non-segmentation tasks."""
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_params = embedding_dim * output_dim + output_dim

    def forward(self, embeddings: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
    """Per-pixel linear prediction head for segmentation (1x1 conv)."""
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_params = embedding_dim * num_classes + num_classes

    def forward(self, embeddings: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        w_size = self.embedding_dim * self.num_classes
        if theta.dim() == 1:
            W = theta[:w_size].view(self.num_classes, self.embedding_dim, 1, 1)
            b = theta[w_size:].view(self.num_classes)
            return F.conv2d(embeddings, W, b)
        elif theta.dim() == 2:
            S = theta.size(0)
            B, D, H, W_spatial = embeddings.shape
            W = theta[:, :w_size].view(S, self.num_classes, self.embedding_dim)
            b = theta[:, w_size:].view(S, self.num_classes)
            emb_flat = embeddings.view(B, D, -1).permute(0, 2, 1)
            results = []
            for s in range(S):
                logits_flat = emb_flat @ W[s].T + b[s]
                logits = logits_flat.permute(0, 2, 1).view(B, self.num_classes, H, W_spatial)
                results.append(logits)
            return torch.stack(results, dim=0)


# =============================================================================
# 4. Inference Network h_γ (unchanged)
# =============================================================================

class InferenceNetwork(nn.Module):
    """Amortized inference network h_γ."""
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
# 5. Adaptive Prior (unchanged)
# =============================================================================

class AdaptivePrior(nn.Module):
    """Energy-based adaptive prior p(θ | x_{1:N}, x*)."""
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

    def compute_energy(self, train_logits, test_logits):
        if self.task == "classification":
            return self._energy_classification(train_logits, test_logits)
        elif self.task == "segmentation":
            return self._energy_segmentation(train_logits, test_logits)
        else:
            return self._energy_regression(train_logits, test_logits)

    def _energy_classification(self, train_logits, test_logits):
        train_log_probs = F.log_softmax(train_logits, dim=-1)
        train_energy = train_log_probs.sum()
        if test_logits.dim() == 1:
            test_logits = test_logits.unsqueeze(0)
        test_log_probs = F.log_softmax(test_logits, dim=-1)
        test_energy = test_log_probs.sum()
        return train_energy + test_energy

    # def _energy_segmentation(self, train_logits, test_logits):
    #     if train_logits.dim() == 4:
    #         train_log_probs = F.log_softmax(train_logits, dim=1)
    #     else:
    #         train_log_probs = F.log_softmax(train_logits, dim=-3)
    #     # train_energy = train_log_probs.sum()
    #     train_energy = train_log_probs.mean()
    #     if test_logits.dim() == 3:
    #         test_logits = test_logits.unsqueeze(0)
    #     test_log_probs = F.log_softmax(test_logits, dim=1)
    #     test_energy = test_log_probs.sum()
    #     # test_energy = test_log_probs.mean()
    #     return train_energy + test_energy
    def _energy_segmentation(self, train_logits, test_logits):
        train_probs = F.softmax(train_logits, dim=1)
        train_log_probs = torch.log(train_probs + 1e-8)
        train_energy = (train_probs * train_log_probs).sum(dim=1).mean(dim=(1, 2)).sum()
        # train_energy = (train_probs * train_log_probs).sum(dim=1).mean() # for debug 

        test_probs = F.softmax(test_logits, dim=1)
        test_log_probs = torch.log(test_probs + 1e-8)
        test_energy = (test_probs * test_log_probs).sum(dim=1).mean(dim=(1, 2)).sum()
        # test_energy = (test_probs * test_log_probs).sum(dim=1).mean()
        
        
        return train_energy + test_energy

    def _energy_regression(self, train_preds, test_preds):
        y_samples = torch.linspace(self.y_min, self.y_max, self.mc_samples, device=train_preds.device)
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

    def log_prior(self, train_logits, test_logits):
        return self.compute_energy(train_logits, test_logits)


# =============================================================================
# 6. ELBO Computation 
# =============================================================================

class ELBOComputer(nn.Module):
    """Computes the ELBO objective."""
    def __init__(self, task: str = "classification", kl_weight: float = 1.0, num_classes: int = 2):
        super().__init__()
        self.task = task
        self.kl_weight = kl_weight
        self.num_classes = num_classes
        self.prior = AdaptivePrior(task=task, num_classes=num_classes)

    def forward(self, train_x_emb, train_y, test_x_emb, theta, mu, log_sigma, prediction_head):
        if self.task == "segmentation":
            return self._forward_segmentation(
                train_x_emb, train_y, test_x_emb, theta, mu, log_sigma, prediction_head
            )
        else:
            return self._forward_standard(
                train_x_emb, train_y, test_x_emb, theta, mu, log_sigma, prediction_head
            )

    def _forward_standard(self, train_x_emb, train_y, test_x_emb, theta, mu, log_sigma, prediction_head):
        train_preds = prediction_head(train_x_emb, theta)
        log_lik = self._log_likelihood(train_preds, train_y)
        test_preds = prediction_head(
            test_x_emb.unsqueeze(0) if test_x_emb.dim() == 1 else test_x_emb, theta,
        )
        log_prior = self.prior.log_prior(train_preds, test_preds)
        sigma = torch.exp(log_sigma)

        log_q_sample = torch.sum(
            -0.5 * ((theta - mu) / sigma) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
        )
        # debug: take mean instead of sum
        # log_q_sample = torch.mean(
        #     -0.5 * ((theta - mu) / sigma) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
        # )
        return log_lik + self.kl_weight * (log_prior - log_q_sample)

    def _forward_segmentation(self, train_x_emb, train_y, test_x_emb, theta, mu, log_sigma, prediction_head):
        train_logits = prediction_head(train_x_emb, theta)
        log_lik = self._log_likelihood_segmentation(train_logits, train_y)
        test_logits = prediction_head(test_x_emb, theta)
        log_prior = self.prior.log_prior(train_logits, test_logits)
        sigma = torch.exp(log_sigma)

        log_q_sample = torch.sum(
            -0.5 * ((theta - mu) / sigma) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
        )
        return log_lik + self.kl_weight * (log_prior - log_q_sample)

    def _log_likelihood(self, preds, targets):
        if self.task == "classification":
            return -F.cross_entropy(preds, targets.long(), reduction="sum")
        else:
            return -0.5 * torch.sum((preds - targets) ** 2)

    def _log_likelihood_segmentation(self, logits, targets):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        # debug: reduction over mean
        # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        batch_size = logits.shape[0]
        logits_flat = logits.view(batch_size, self.num_classes, -1)
        targets_flat = targets.view(batch_size, -1).long()
        # return -F.cross_entropy(logits, targets.long(), reduction="mean")
        # return -torch.mean(
        #     torch.sum(F.cross_entropy(logits_flat, targets_flat), dim=1)
        # )
        return -torch.mean(
            torch.sum(criterion(target=targets_flat, input=logits_flat), dim=1)
        )
        # criterion = nn.CrossEntropyLoss(reduction='mean')
        # batch_size = logits.shape[0]
        # logits_flat = logits.view(batch_size, self.num_classes, -1)
        # targets_flat = targets.view(batch_size, -1).long()
        # return -criterion(input=logits_flat, target=targets_flat)


# =============================================================================
# 7. Synthetic Environment Generator (unchanged)
# =============================================================================

class SyntheticEnvironmentGenerator:
    """Generates synthetic environments by bootstrap subsampling."""
    def __init__(self, train_x, train_y, n_train, n_test):
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
# 8. VIDS Lightning Module
# =============================================================================

class VIDS(L.LightningModule):
    """
    Variational Inference under Distribution Shifts (VIDS).
    PyTorch Lightning implementation.
    
    Supports: 'classification', 'regression', 'segmentation'
    
    Only the inference network is trained; the embedding network is frozen.
    Logging only happens during VIDS training (not during pre-training).
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
        # Training hyperparameters
        learning_rate: float = 1e-3,
        num_environments: int = 10,
        env_train_size: int = 500,
        env_test_size: int = 20,
        # Prediction hyperparameters
        num_prediction_samples: int = 100,
        summary_stats = 'mean_std'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_net"])

        self.embedding_net = embedding_net
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.task = task
        self.variance_penalty = variance_penalty
        self.lr = learning_rate
        self.num_environments = num_environments
        self.env_train_size = env_train_size
        self.env_test_size = env_test_size
        self.num_prediction_samples = num_prediction_samples
        self.summary_stats = summary_stats
        

        # code for summary dim if we want richer summary statistics
        if task == "segmentation" and summary_stats == "mean_std":
            self.summary_dim = 2 * embedding_dim  # mean + std
        else:
            self.summary_dim = embedding_dim


        # Choose appropriate prediction head
        if task == "segmentation":
            self.prediction_head = SegmentationPredictionHead(embedding_dim, output_dim)
        else:
            self.prediction_head = PredictionHead(embedding_dim, output_dim)

        theta_dim = self.prediction_head.num_params

        # Inference network (the only trainable part)
        self.inference_net = InferenceNetwork(
            # embedding_dim=embedding_dim,
            embedding_dim=self.summary_dim,
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

        # Freeze prediction head (its params are not directly optimized;
        # θ is sampled via the inference network)
        for param in self.prediction_head.parameters():
            param.requires_grad = False

    # -----------------------------------------------------------------
    # Embedding helpers
    # -----------------------------------------------------------------

    @torch.no_grad()
    def compute_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Compute embeddings using the frozen pre-trained network."""
        self.embedding_net.eval()
        return self.embedding_net(x)

    # def aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
    #     """Aggregate embeddings into a single summary vector."""
    #     if self.task == "segmentation":
    #         pooled = embeddings.mean(dim=(2, 3))
    #         return pooled.mean(dim=0)
    #     else:
    #         return embeddings.mean(dim=0)

    # def aggregate_single_image_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
    #     """Spatially pool a single image's dense embedding to a global vector."""
    #     if embedding.dim() == 4:
    #         return embedding.mean(dim=(0, 2, 3))
    #     elif embedding.dim() == 3:
    #         return embedding.mean(dim=(1, 2))
    #     else:
    #         return embedding
    def aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate a batch of embeddings into a single summary vector.

        For non-segmentation: embeddings is (B, D)
            mean → (D,)

        For segmentation: embeddings is (B, D, H, W)
            "mean"     → spatial-pool each image to (B, D), then mean over B → (D,)
            "mean_std" → spatial-pool each image to (B, D), compute mean
                         and std over B, concatenate → (2D,)
        """
        if self.task == "segmentation":
            # Pool each image spatially: (B, D, H, W) → (B, D)
            pooled = embeddings.mean(dim=(2, 3))
            # Aggregate across images in the environment
            batch_mean = pooled.mean(dim=0)  # (D,)

            if self.summary_stats == "mean_std":
                if pooled.size(0) > 1:
                    batch_std = pooled.std(dim=0)  # (D,)
                else:
                    batch_std = torch.zeros_like(batch_mean)
                return torch.cat([batch_mean, batch_std], dim=0)  # (2D,)
            else:
                return batch_mean  # (D,)
        else:
            return embeddings.mean(dim=0)  # (D,)

    def aggregate_single_image_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Spatially pool a single test image's dense embedding to a summary vector.

        Input shapes:
            (B, D, H, W) with B=1, or (D, H, W), or (D,)

        Output:
            (D,) if summary_stats == "mean"
            (2D,) if summary_stats == "mean_std"
        """
        if embedding.dim() == 4:
            # (1, D, H, W) → pool over H, W
            pooled = embedding.mean(dim=(2, 3))  # (1, D)
            spatial_mean = pooled.squeeze(0)       # (D,)

            if self.task == "segmentation" and self.summary_stats == "mean_std":
                # For a single image, compute std over spatial locations
                # embedding is (1, D, H, W) → (D, H*W)
                flat = embedding.squeeze(0).flatten(1)  # (D, H*W)
                spatial_std = flat.std(dim=1)            # (D,)
                return torch.cat([spatial_mean, spatial_std], dim=0)  # (2D,)
            else:
                return spatial_mean

        elif embedding.dim() == 3:
            # (D, H, W)
            spatial_mean = embedding.mean(dim=(1, 2))  # (D,)

            if self.task == "segmentation" and self.summary_stats == "mean_std":
                flat = embedding.flatten(1)      # (D, H*W)
                spatial_std = flat.std(dim=1)    # (D,)
                return torch.cat([spatial_mean, spatial_std], dim=0)  # (2D,)
            else:
                return spatial_mean
        else:
            # Already a vector (D,)
            if self.task == "segmentation" and self.summary_stats == "mean_std":
                # No spatial info available, pad with zeros
                return torch.cat([embedding, torch.zeros_like(embedding)], dim=0)
            return embedding

    def sample_theta(self, mu, log_sigma):
        """Reparameterization trick."""
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    # -----------------------------------------------------------------
    # Core forward / loss computation
    # -----------------------------------------------------------------

    def forward(self, train_x: torch.Tensor, test_x: torch.Tensor, num_samples: Optional[int] = None):
        """
        Inference forward pass — calls predict internally.
        Useful for Lightning's predict step and manual inference.
        """
        if num_samples is None:
            num_samples = self.num_prediction_samples
        return self.predict_with_uncertainty(train_x, test_x, num_samples)

    def compute_environment_elbo(
        self,
        env_train_x: torch.Tensor,
        env_train_y: torch.Tensor,
        env_test_x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ELBO for a single synthetic environment."""
        train_emb = self.compute_embeddings(env_train_x)
        test_emb = self.compute_embeddings(env_test_x)
        train_summary = self.aggregate_embeddings(train_emb)

        total_elbo = torch.tensor(0.0, device=env_train_x.device)

        if self.task == "segmentation":
            for j in range(env_test_x.size(0)):
                test_emb_j = test_emb[j:j+1]
                test_summary_j = self.aggregate_single_image_embedding(test_emb_j)
                mu, log_sigma = self.inference_net(train_summary, test_summary_j)
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
        else:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute full VIDS training loss.
        
        Returns:
            loss: scalar total loss
            mean_elbo: scalar mean ELBO across environments
            var_penalty: scalar variance penalty
        """
        batch_size = train_x.size(0)

        if batch_size < env_test_size:
            real_test_size = max(1, batch_size // 2)
            real_train_size = max(1, batch_size - real_test_size)
        else:
            real_test_size = env_test_size
            real_train_size = env_train_size

        env_generator = SyntheticEnvironmentGenerator(
            train_x, train_y, real_train_size, real_test_size
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
        var_penalty = env_elbos.var() if num_environments > 1 else torch.tensor(0.0, device=train_x.device)
        loss = -mean_elbo + self.variance_penalty * var_penalty
        return loss, mean_elbo, var_penalty

    # -----------------------------------------------------------------
    # Lightning training / validation steps
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss, mean_elbo, var_penalty = self.compute_loss(
            train_x=x,
            train_y=y,
            num_environments=self.num_environments,
            env_train_size=self.env_train_size,
            env_test_size=self.env_test_size,
        )

        # Logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mean_elbo", mean_elbo, on_step=False, on_epoch=True)
        self.log("train_var_penalty", var_penalty, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        loss, mean_elbo, var_penalty = self.compute_loss(
            train_x=x,
            train_y=y,
            num_environments=self.num_environments,
            env_train_size=self.env_train_size,
            env_test_size=self.env_test_size,
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mean_elbo", mean_elbo, on_step=False, on_epoch=True)
        self.log("val_var_penalty", var_penalty, on_step=False, on_epoch=True)

        # Log images for first batch
        if batch_idx == 0 and self.task == "segmentation":
            self._log_segmentation_images(batch, stage="val")

        return loss

    # -----------------------------------------------------------------
    # Image logging
    # -----------------------------------------------------------------

    def _create_overlay_image(self, image, mask, alpha=0.5):
        """Create an RGB overlay of a mask on a grayscale/RGB image."""
        if image.shape[0] == 1:
            image_rgb = image.repeat(3, 1, 1)
        else:
            image_rgb = image.clone()

        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0

        colored_mask = torch.zeros(3, mask.shape[0], mask.shape[1], device=mask.device)
        colored_mask[0][mask == 1] = 1.0  # Red for class 1

        overlay = (1 - alpha) * image_rgb + alpha * colored_mask
        overlay = torch.clamp(overlay, 0, 1)
        return overlay

    @torch.no_grad()
    def _log_segmentation_images(self, batch, stage="val"):
        """Log input / ground truth / predicted mask overlays."""
        if not self.logger:
            return

        x, y = batch
        # Use the batch itself as 'training context' for prediction
        results = self.predict_with_uncertainty(
            train_x=x, test_x=x[:1], num_samples=min(20, self.num_prediction_samples)
        )

        img = x[0].cpu()
        gt_mask = y[0].cpu()
        pred_mask = results["predicted_masks"][0].cpu()

        gt_overlay = self._create_overlay_image(img, gt_mask)
        pred_overlay = self._create_overlay_image(img, pred_mask)

        # Normalise uncertainty map to [0, 1] for visualisation
        unc_map = results["uncertainty_map"][0].cpu()
        if unc_map.max() > 0:
            unc_map = unc_map / unc_map.max()
        unc_image = unc_map.unsqueeze(0).repeat(3, 1, 1)  # grayscale → RGB

        self.logger.experiment.add_image(f"{stage}/input", img, self.current_epoch)
        self.logger.experiment.add_image(f"{stage}/ground_truth", gt_overlay, self.current_epoch)
        self.logger.experiment.add_image(f"{stage}/prediction", pred_overlay, self.current_epoch)
        self.logger.experiment.add_image(f"{stage}/uncertainty", unc_image, self.current_epoch)

    # -----------------------------------------------------------------
    # Prediction with uncertainty
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        train_x: torch.Tensor,
        test_x: torch.Tensor,
        num_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation.
        This is the main inference entry point (replaces the old `predict`).
        """
        was_training = self.training
        self.eval()

        train_emb = self.compute_embeddings(train_x)
        test_emb = self.compute_embeddings(test_x)
        train_summary = self.aggregate_embeddings(train_emb)

        if self.task == "segmentation":
            result = self._predict_segmentation(train_summary, test_emb, num_samples)
        else:
            result = self._predict_standard(train_summary, test_emb, test_x, num_samples)

        if was_training:
            self.train()
        return result

    def _predict_standard(self, train_summary, test_emb, test_x, num_samples):
        M = test_x.size(0)
        all_predictions = []
        for j in range(M):
            test_emb_j = test_emb[j]
            mu, log_sigma = self.inference_net(train_summary, test_emb_j)
            sigma = torch.exp(log_sigma)
            eps = torch.randn(num_samples, mu.size(0), device=mu.device)
            theta_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
            preds = self.prediction_head(test_emb_j.unsqueeze(0), theta_samples)
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
        M = test_emb.size(0)
        all_probs = []

        for j in range(M):
            test_emb_j = test_emb[j:j+1]
            test_summary_j = self.aggregate_single_image_embedding(test_emb_j)

            mu, log_sigma = self.inference_net(train_summary, test_summary_j)
            sigma = torch.exp(log_sigma)
            eps = torch.randn(num_samples, mu.size(0), device=mu.device)
            theta_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

            logits = self.prediction_head(test_emb_j, theta_samples)
            probs = F.softmax(logits, dim=2)
            all_probs.append(probs.squeeze(1))

        all_probs = torch.stack(all_probs, dim=1)

        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        predicted_masks = mean_probs.argmax(dim=1)

        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
        per_sample_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2)
        expected_entropy = per_sample_entropy.mean(dim=0)
        mutual_info = entropy - expected_entropy

        return {
            "predictions": mean_probs,
            "std": std_probs,
            "predicted_masks": predicted_masks,
            "uncertainty_map": entropy,
            "aleatoric_uncertainty": expected_entropy,
            "epistemic_uncertainty": mutual_info,
            "samples": all_probs,
        }

    # -----------------------------------------------------------------
    # Optimizer — only trains the inference network
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.inference_net.parameters(), lr=self.lr)
        return optimizer




class PretrainSegmentationEmbedding(L.LightningModule):
    """
    Lightning module that wraps UNetDenseEmbedding + a 1x1 conv head
    for pre-training the embedding network on segmentation.
    
    After training, extract `self.embedding_net` and freeze it for VIDS.
    """

    def __init__(
        self,
        embedding_net: UNetDenseEmbedding,
        num_classes: int = 2,
        lr: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        dice_alpha: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_net", "class_weights"])

        self.embedding_net = embedding_net
        self.pretrain_head = nn.Conv2d(
            embedding_net.embedding_dim, num_classes, kernel_size=1
        )
        self.num_classes = num_classes
        self.lr = lr
        self.class_weights = class_weights
        self.dice_alpha = dice_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_net(x)
        logits = self.pretrain_head(embeddings)
        return logits

    # -----------------------------------------------------------------
    # Loss & metrics (mirroring your UNet)
    # -----------------------------------------------------------------

    def loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        # Cross-Entropy
        ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean")
        ce_loss = ce_loss_fn(pred, target.long())

        # Dice Loss
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
        union = torch.sum(pred_soft + target_one_hot, dim=(2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score.mean()# * 0

        return self.dice_alpha * ce_loss + (1 - self.dice_alpha) * dice_loss

    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
        union = torch.sum(pred_soft + target_one_hot, dim=(2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        return dice_score.mean()

    # -----------------------------------------------------------------
    # Image logging (mirroring your UNet)
    # -----------------------------------------------------------------

    def _create_overlay_image(self, image, mask, alpha=0.5):
        if image.shape[0] == 1:
            image_rgb = image.repeat(3, 1, 1)
        else:
            image_rgb = image.clone()

        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0

        colored_mask = torch.zeros(3, mask.shape[0], mask.shape[1], device=mask.device)
        colored_mask[0][mask == 1] = 1.0  # Red for class 1

        overlay = (1 - alpha) * image_rgb + alpha * colored_mask
        return torch.clamp(overlay, 0, 1)

    def _log_images(self, batch, pred, stage="train"):
        if not self.logger:
            return

        x, y = batch
        img = x[0].cpu()
        gt_mask = y[0].cpu()
        pred_mask = torch.argmax(torch.softmax(pred[0], dim=0), dim=0).cpu()

        gt_overlay = self._create_overlay_image(img, gt_mask)
        pred_overlay = self._create_overlay_image(img, pred_mask)

        self.logger.experiment.add_image(
            f"{stage}/input_image", img, self.current_epoch
        )
        self.logger.experiment.add_image(
            f"{stage}/ground_truth_overlay", gt_overlay, self.current_epoch
        )
        self.logger.experiment.add_image(
            f"{stage}/prediction_overlay", pred_overlay, self.current_epoch
        )

    # -----------------------------------------------------------------
    # Lightning steps
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        dice = self.dice_coefficient(pred, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._log_images(batch, pred, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        dice = self.dice_coefficient(pred, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._log_images(batch, pred, stage="val")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def make_prediction(self, img):
        out = self.forward(img)
        return torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze()

    # -----------------------------------------------------------------
    # Convenience: freeze and return the embedding net
    # -----------------------------------------------------------------

    def get_frozen_embedding_net(self) -> UNetDenseEmbedding:
        """Call after training to get the frozen embedding network for VIDS."""
        self.embedding_net.eval()
        for param in self.embedding_net.parameters():
            param.requires_grad = False
        return self.embedding_net