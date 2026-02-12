"""
VIDS: Variational Inference under Distribution Shifts
PyTorch Implementation

Based on: "Quantifying Uncertainty in the Presence of Distribution Shifts"
by Slavutsky & Blei (NeurIPS 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict
import math


# =============================================================================
# 1. Embedding Networks (Pre-trained feature extractors)
# =============================================================================

class FCEmbedding(nn.Module):
    """Fully connected embedding network for tabular data."""

    def __init__(self, input_dim: int, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim)
            # nn.ReLU(),
            # nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvEmbedding(nn.Module):
    """Convolutional embedding network for image data.
    Two conv blocks with 32 filters each, followed by FC layers.
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 16):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = None  # Lazily initialized
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


# =============================================================================
# 2. Prediction Head (the θ-parameterized layer)
# =============================================================================

class PredictionHead(nn.Module):
    """
    Linear prediction head: f_θ(g(x)) = θ^T g(x) (+ bias).
    For classification: returns logits.
    For regression: returns predicted mean.

    This is the layer whose weights θ are treated as random variables
    in the Bayesian framework.
    """

    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        # Total number of parameters: weight matrix + bias
        self.num_params = embedding_dim * output_dim + output_dim

    def forward(
        self, embeddings: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embedding_dim)
            theta: (num_params,) or (num_samples, num_params)

        Returns:
            predictions: (batch, output_dim) or (num_samples, batch, output_dim)
        """
        w_size = self.embedding_dim * self.output_dim

        if theta.dim() == 1:
            # Single theta
            W = theta[:w_size].view(self.embedding_dim, self.output_dim)
            b = theta[w_size:]
            return embeddings @ W + b
        else:
            # Multiple theta samples: (S, num_params)
            S = theta.size(0)
            W = theta[:, :w_size].view(S, self.embedding_dim, self.output_dim)
            b = theta[:, w_size:].unsqueeze(1)  # (S, 1, output_dim)
            # embeddings: (B, d) -> (1, B, d)
            emb = embeddings.unsqueeze(0)
            return torch.bmm(emb.expand(S, -1, -1), W) + b  # (S, B, output_dim)


# =============================================================================
# 3. Inference Network h_γ
# =============================================================================

class InferenceNetwork(nn.Module):
    """
    Amortized inference network h_γ.
    Takes concatenated [g_bar(x_{1:n}), g(x*)] and outputs
    variational parameters φ = (μ, log_σ) of q_φ(θ | x_{1:n}, x*).
    """

    def __init__(
        self,
        embedding_dim: int,
        theta_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        input_dim = 2 * embedding_dim  # concatenation of train summary + test embedding

        if hidden_dims is None:
            # Default: 6 layers as in the paper for synthetic experiments
            d = theta_dim
            hidden_dims = [64 * d, 32 * d, 16 * d, 8 * d, 4 * d]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)
        # Output: mean and log-std for diagonal Gaussian
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
        # Expand train summary to match test batch
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
# 4. Adaptive Prior (Energy-based prior)
# =============================================================================

class AdaptivePrior(nn.Module):
    """
    Energy-based adaptive prior p(θ | x_{1:N}, x*).

    E(θ; x_{1:N}, x*) = ∫ [Σ_i log p(y|x_i, θ) + log p(y|x*, θ)] dy

    For classification (discrete Y): summation over classes.
    For regression (continuous Y): Monte Carlo integration.
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
        assert task in ["classification", "regression"]
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
        """
        Compute the energy E(θ; x_{1:N}, x*).

        Args:
            train_logits: (N, output_dim) - predictions on training covariates
            test_logits: (output_dim,) or (M, output_dim) - predictions on test covariates

        Returns:
            energy: scalar
        """
        if self.task == "classification":
            return self._energy_classification(train_logits, test_logits)
        else:
            return self._energy_regression(train_logits, test_logits)

    def _energy_classification(
        self, train_logits: torch.Tensor, test_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        For binary/multi-class: sum over all classes y of log p(y|x, θ).
        E = Σ_i Σ_y log p(y|x_i, θ) + Σ_y log p(y|x*, θ)
        """
        # log p(y|x, θ) for all classes = log_softmax
        train_log_probs = F.log_softmax(train_logits, dim=-1)  # (N, C)
        # Sum over classes and data points
        train_energy = train_log_probs.sum()

        if test_logits.dim() == 1:
            test_logits = test_logits.unsqueeze(0)
        test_log_probs = F.log_softmax(test_logits, dim=-1)  # (M, C)
        test_energy = test_log_probs.sum()

        return train_energy + test_energy

    def _energy_regression(
        self, train_preds: torch.Tensor, test_preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Monte Carlo approximation for continuous Y.
        Sample r values uniformly from [y_min, y_max] and compute
        log-likelihood under unit-variance Gaussian.
        """
        # Sample y values
        y_samples = torch.linspace(
            self.y_min, self.y_max, self.mc_samples, device=train_preds.device
        )

        # train_preds: (N, 1) -> predicted means
        # log p(y|x, θ) = -0.5 * (y - μ)^2 - 0.5 * log(2π)
        # For each y_sample, compute across all training points
        train_mu = train_preds.squeeze(-1)  # (N,)
        # (mc_samples, N)
        diff_train = y_samples.unsqueeze(1) - train_mu.unsqueeze(0)
        log_lik_train = -0.5 * diff_train**2 - 0.5 * math.log(2 * math.pi)
        # Sum over training points, then sum over y samples (MC integration)
        # Scale by (y_max - y_min) / mc_samples for proper MC estimate
        scale = (self.y_max - self.y_min) / self.mc_samples
        train_energy = (log_lik_train.sum(dim=1) * scale).sum()

        if test_preds.dim() == 1:
            test_preds = test_preds.unsqueeze(0)
        test_mu = test_preds.squeeze(-1)  # (M,)
        diff_test = y_samples.unsqueeze(1) - test_mu.unsqueeze(0)
        log_lik_test = -0.5 * diff_test**2 - 0.5 * math.log(2 * math.pi)
        test_energy = (log_lik_test.sum(dim=1) * scale).sum()

        return train_energy + test_energy

    def log_prior(
        self,
        train_logits: torch.Tensor,
        test_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unnormalized log prior: log p(θ|x_{1:N}, x*) ∝ E(θ; x_{1:N}, x*).
        The normalizing constant Z(θ) is intractable, so we work with
        the unnormalized version (sufficient for the KL in ELBO).
        """
        return self.compute_energy(train_logits, test_logits)


# =============================================================================
# 5. ELBO Computation
# =============================================================================

class ELBOComputer(nn.Module):
    """
    Computes the ELBO objective:
    L(φ; x*, D) = E_q[log p(y_{1:N}|x_{1:N}, θ)] - KL(q_φ(θ; x*) || p(θ|x_{1:N}, x*))

    Since p(θ|x_{1:N}, x*) is only known up to a normalizing constant,
    we use the form:
    L = E_q[log p(y_{1:N}|x_{1:N}, θ)] + E_q[log p(θ|x_{1:N}, x*)] - E_q[log q_φ(θ)]
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
        prediction_head: PredictionHead,
    ) -> torch.Tensor:
        """
        Args:
            train_x_emb: (N, d) embedded training covariates
            train_y: (N,) training labels/targets
            test_x_emb: (d,) embedded test covariate
            theta: (theta_dim,) sampled parameters
            mu: (theta_dim,) variational mean
            log_sigma: (theta_dim,) variational log std
            prediction_head: the prediction head module

        Returns:
            elbo: scalar (to be maximized)
        """
        # 1. Log-likelihood: log p(y_{1:N} | x_{1:N}, θ)
        train_preds = prediction_head(train_x_emb, theta)  # (N, output_dim)
        log_lik = self._log_likelihood(train_preds, train_y)

        # 2. Log prior: log p(θ | x_{1:N}, x*)
        test_preds = prediction_head(
            test_x_emb.unsqueeze(0) if test_x_emb.dim() == 1 else test_x_emb,
            theta,
        )
        log_prior = self.prior.log_prior(train_preds, test_preds)

        # 3. Log q: entropy of the variational distribution
        sigma = torch.exp(log_sigma)
        log_q = -0.5 * torch.sum(
            1.0 + 2.0 * log_sigma + math.log(2 * math.pi)
        )  # Negative entropy (log q evaluated at theta = mu + sigma * eps)
        # More precisely, for a single sample:
        log_q_sample = torch.sum(
            -0.5 * ((theta - mu) / sigma) ** 2
            - log_sigma
            - 0.5 * math.log(2 * math.pi)
        )

        # ELBO = E_q[log p(y|x,θ)] + E_q[log p(θ|x,x*)] - E_q[log q(θ)]
        # Single sample estimate:
        elbo = log_lik + self.kl_weight * (log_prior - log_q_sample)

        return elbo

    def _log_likelihood(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.task == "classification":
            return -F.cross_entropy(preds, targets.long(), reduction="sum")
        else:
            # Gaussian likelihood with unit variance
            return -0.5 * torch.sum((preds - targets) ** 2)


# =============================================================================
# 6. Synthetic Environment Generator
# =============================================================================

class SyntheticEnvironmentGenerator:
    """
    Generates synthetic environments by bootstrap subsampling from training data.
    Each environment consists of a (D_tr, D_te) pair with potentially different
    empirical distributions, simulating covariate shifts.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        n_train: int,
        n_test: int,
    ):
        """
        Args:
            train_x: (N, ...) full training covariates
            train_y: (N,) full training targets
            n_train: size of each synthetic training set
            n_test: size of each synthetic test set
        """
        self.train_x = train_x
        self.train_y = train_y
        self.N = train_x.size(0)
        self.n_train = n_train
        self.n_test = n_test

    def sample_environment(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one synthetic environment by bootstrap sampling.

        Returns:
            env_train_x, env_train_y, env_test_x, env_test_y
        """
        # Sample training subset
        train_idx = torch.randint(0, self.N, (self.n_train,))
        # Sample test subset (independent bootstrap)
        test_idx = torch.randint(0, self.N, (self.n_test,))

        return (
            self.train_x[train_idx],
            self.train_y[train_idx],
            self.train_x[test_idx],
            self.train_y[test_idx],
        )


# =============================================================================
# 7. VIDS Model (Main class)
# =============================================================================

class VIDS(nn.Module):
    """
    Variational Inference under Distribution Shifts (VIDS).

    Combines:
    1. Pre-trained embedding network g_ξ
    2. Prediction head f_θ
    3. Inference network h_γ (amortized variational posterior)
    4. Adaptive energy-based prior
    5. Synthetic environment training
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
            embedding_dim: dimension of embeddings
            output_dim: number of output classes (classification) or 1 (regression)
            task: 'classification' or 'regression'
            inference_hidden_dims: hidden layer sizes for h_γ
            kl_weight: weight λ for KL term in ELBO
            variance_penalty: τ for cross-environment variance penalty
            num_classes: number of classes for classification
        """
        super().__init__()

        self.embedding_net = embedding_net
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.task = task
        self.variance_penalty = variance_penalty

        # Prediction head
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

        # Freeze embedding network (pre-trained)
        for param in self.embedding_net.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def compute_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Compute embeddings using the frozen pre-trained network."""
        self.embedding_net.eval()
        return self.embedding_net(x)

    def aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate training embeddings into a single summary statistic (mean).
        Follows the Deep Sets approach.
        """
        return embeddings.mean(dim=0)

    def sample_theta(
        self, mu: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: θ = μ + σ ⊙ ε"""
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

        Args:
            env_train_x: (n, ...) training covariates
            env_train_y: (n,) training targets
            env_test_x: (m, ...) test covariates

        Returns:
            total_elbo: sum of ELBOs over test points
        """
        # Compute embeddings
        train_emb = self.compute_embeddings(env_train_x)  # (n, d)
        test_emb = self.compute_embeddings(env_test_x)  # (m, d)

        # Aggregate training embeddings
        train_summary = self.aggregate_embeddings(train_emb)  # (d,)

        total_elbo = torch.tensor(0.0, device=env_train_x.device)

        # For each test point
        for j in range(env_test_x.size(0)):
            test_emb_j = test_emb[j]  # (d,)

            # Get variational parameters from inference network
            mu, log_sigma = self.inference_net(train_summary, test_emb_j)

            # Sample θ using reparameterization trick
            theta = self.sample_theta(mu, log_sigma)

            # Compute ELBO
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
        Compute the full VIDS training loss with cross-environment variance penalty.

        L = Σ_ℓ L^(ℓ) + τ * Var(L^(1), ..., L^(L))

        Args:
            train_x: (N, ...) full training data
            train_y: (N,) full training targets
            num_environments: L - number of synthetic environments
            env_train_size: n - size of each environment's training set
            env_test_size: m - size of each environment's test set

        Returns:
            loss: scalar (to be minimized, so we negate the ELBO)
        """
        env_generator = SyntheticEnvironmentGenerator(
            train_x, train_y, env_train_size, env_test_size
        )

        env_elbos = []
        for _ in range(num_environments):
            (
                env_train_x,
                env_train_y,
                env_test_x,
                env_test_y,
            ) = env_generator.sample_environment()

            elbo = self.compute_environment_elbo(
                env_train_x, env_train_y, env_test_x
            )
            env_elbos.append(elbo)

        env_elbos = torch.stack(env_elbos)

        # Cross-environment objective (Eq. 12)
        mean_elbo = env_elbos.mean()
        var_penalty = env_elbos.var() if num_environments > 1 else torch.tensor(0.0)

        # Minimize negative ELBO + variance penalty
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
        Predict with uncertainty estimation for test inputs.

        Args:
            train_x: (N, ...) training covariates (for computing train summary)
            test_x: (M, ...) test covariates
            num_samples: S - number of posterior samples

        Returns:
            dict with:
                'predictions': (M, output_dim) mean predictions
                'std': (M, output_dim) standard deviations
                'samples': (S, M, output_dim) all prediction samples
        """
        self.eval()

        # Compute embeddings
        train_emb = self.compute_embeddings(train_x)
        test_emb = self.compute_embeddings(test_x)

        # Aggregate training embeddings
        train_summary = self.aggregate_embeddings(train_emb)

        M = test_x.size(0)
        all_predictions = []

        for j in range(M):
            test_emb_j = test_emb[j]

            # Get variational parameters
            mu, log_sigma = self.inference_net(train_summary, test_emb_j)

            # Draw S samples from the variational posterior
            sigma = torch.exp(log_sigma)
            eps = torch.randn(num_samples, mu.size(0), device=mu.device)
            theta_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, theta_dim)

            # Compute predictions for each sample
            preds = self.prediction_head(
                test_emb_j.unsqueeze(0), theta_samples
            )  # (S, 1, output_dim)
            all_predictions.append(preds.squeeze(1))  # (S, output_dim)

        # Stack: (S, M, output_dim)
        all_predictions = torch.stack(all_predictions, dim=1)

        if self.task == "classification":
            # Apply softmax to get probabilities
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


# =============================================================================
# 8. Pre-training utilities
# =============================================================================

# def pretrain_embedding(
#     embedding_net: nn.Module,
#     prediction_head_pretrain: nn.Module,
#     train_x: torch.Tensor,
#     train_y: torch.Tensor,
#     task: str = "classification",
#     epochs: int = 100,
#     lr: float = 1e-3,
#     batch_size: int = 64,
# ) -> nn.Module:
#     """
#     Pre-train the embedding network g_ξ to maximize p(y|x, θ) = f_θ(g_ξ(x)).

#     Args:
#         embedding_net: the embedding network to pre-train
#         prediction_head_pretrain: a standard nn.Linear head for pre-training
#         train_x: training covariates
#         train_y: training targets
#         task: 'classification' or 'regression'
#         epochs: number of pre-training epochs
#         lr: learning rate
#         batch_size: batch size

#     Returns:
#         Pre-trained embedding network (frozen after this)
#     """
#     device = next(embedding_net.parameters()).device

#     model = nn.Sequential(embedding_net, prediction_head_pretrain).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     dataset = torch.utils.data.TensorDataset(train_x, train_y)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for x_batch, y_batch in loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()

#             preds = model(x_batch)
#             if task == "classification":
#                 loss = F.cross_entropy(preds, y_batch.long())
#             else:
#                 loss = F.mse_loss(preds.squeeze(-1), y_batch)

#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         if (epoch + 1) % 20 == 0:
#             print(f"  Pre-train Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

#     # Freeze embedding network
#     embedding_net.eval()
#     for param in embedding_net.parameters():
#         param.requires_grad = False

#     return embedding_net

def pretrain_embedding(
    embedding_net, prediction_head_pretrain,
    train_x, train_y,
    task="regression", epochs=100, lr=1e-3, batch_size=64,
):
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
            else:
                loss = F.mse_loss(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Pre-train Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Verify final performance
    model.eval()
    with torch.no_grad():
        final_preds = model(train_x)
        final_mse = F.mse_loss(final_preds, train_y)
        print(f"  Final train MSE: {final_mse.item():.6f}")

    embedding_net.eval()
    for param in embedding_net.parameters():
        param.requires_grad = False

    return embedding_net, model


# =============================================================================
# 9. Training Loop
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

    Args:
        model: VIDS model
        train_x: (N, ...) training covariates
        train_y: (N,) training targets
        num_environments: L - number of synthetic environments per step
        env_train_size: n - training set size per environment
        env_test_size: m - test set size per environment
        num_steps: K - number of optimization steps
        lr: learning rate η
        verbose: whether to print progress

    Returns:
        losses: list of loss values
    """
    # Only optimize inference network parameters
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