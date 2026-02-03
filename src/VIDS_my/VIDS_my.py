import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

# ---------------------------------------------------------
# 1. Dataset Generation (User Provided Logic)
# ---------------------------------------------------------
# Parameters based on Paper Section 4.1.1
N = 500
a = 0.5
b = 1.0
dtype = np.float32

# Generate Data
Xs = np.sort(np.random.uniform(low=0.0, high=a, size=N)).astype(dtype).reshape(-1,1)
new_Xs = np.sort(np.random.uniform(low=0.0, high=b, size=N)).astype(dtype).reshape(-1,1)

v = 0.1 * Xs
new_v = 0.1 * new_Xs

# Epsilon with correct dimensionality handling
epsilon = np.array([np.random.normal(0, v[i].item()) for i in range(N)]).astype(dtype)
new_epsilon = np.array([np.random.normal(0, new_v[i].item()) for i in range(N)]).astype(dtype)

beta = 1.0
Ys = beta * Xs + epsilon.reshape(-1,1)
new_Ys = beta * new_Xs + new_epsilon.reshape(-1,1)

# Convert to Tensors
x_train_tensor = torch.from_numpy(Xs)
y_train_tensor = torch.from_numpy(Ys)
x_test_tensor = torch.from_numpy(new_Xs)
y_test_tensor = torch.from_numpy(new_Ys)

# ---------------------------------------------------------
# 2. VIDS Model Architecture
# ---------------------------------------------------------
class VIDS(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim=64, task='regression'):
        super().__init__()
        self.task = task
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        
        # Prediction Head Parameters: Weights + Bias
        self.num_theta_params = embedding_dim * output_dim + output_dim 
        
        # Inference Network (h_gamma)
        # Inputs: Aggregated Train Embeddings (dim) + Test Embedding (dim)
        self.inference_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_theta_params * 2) # Outputs mu and rho
        )

    def sample_theta(self, mu, log_sigma):
        # Reparameterization trick
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        theta_flat = mu + std * eps
        
        # Reshape to (Batch_M, Out, Dim) and (Batch_M, Out)
        w_end = self.embedding_dim * self.output_dim
        weight = theta_flat[:, :w_end].reshape(-1, self.output_dim, self.embedding_dim)
        bias = theta_flat[:, w_end:]
        return weight, bias, theta_flat

    def predict_with_theta(self, embeddings, weight, bias):
        """
        Broadcasting prediction for ELBO calculation.
        Evaluates M different models (weight/bias) on N different data points (embeddings).
        """
        # weight: (M, Out, Dim)
        # embeddings: (N, Dim)
        # Output: (M, N, Out)
        y_logits = torch.einsum('mod,nd->mno', weight, embeddings)
        y_logits = y_logits + bias.unsqueeze(1)
        return y_logits

    def compute_energy_prior(self, x_train, x_test, weight, bias, y_range=(-3,3)):
        """
        Calculates Adaptive Prior Energy E(theta).
        Includes integration over training data and the specific test point.
        """
        r_samples = 10
        y_samples = torch.FloatTensor(r_samples, 1).uniform_(y_range[0], y_range[1]).to(x_train.device)
        
        # --- 1. Integral over Training Data (Sum_i Integral_y) ---
        # Evaluate M models on N train points
        logits_train = self.predict_with_theta(x_train, weight, bias) # (M, N, 1)
        
        term_train = 0
        for y_s in y_samples:
            # Assuming unit variance Gaussian for integration as per paper/standard practice
            dist = Normal(logits_train, 1.0) 
            log_p = dist.log_prob(y_s) # (M, N, 1)
            term_train += log_p.sum(dim=1).squeeze(-1) # Sum over N -> (M,)
        term_train /= r_samples
        
        # --- 2. Integral over Test Data (Integral_y) ---
        # Evaluate M models on their specific M test points (Diagonal)
        logits_test = torch.einsum('mod,md->mo', weight, x_test) + bias # (M, 1)
        
        term_test = 0
        for y_s in y_samples:
            dist = Normal(logits_test, 1.0)
            term_test += dist.log_prob(y_s).squeeze(-1) # (M,)
        term_test /= r_samples

        return term_train + term_test

    def forward(self, train_embeddings_agg, test_embeddings):
        # Concatenate summary of train + specific test point
        inp = torch.cat([train_embeddings_agg, test_embeddings], dim=1)
        phi = self.inference_net(inp)
        
        mu, rho = torch.chunk(phi, 2, dim=1)
        log_sigma = torch.clamp(rho, min=-5, max=2)
        
        weight, bias, theta_flat = self.sample_theta(mu, log_sigma)
        return weight, bias, mu, log_sigma

# ---------------------------------------------------------
# 3. Training Step with Synthetic Environments
# ---------------------------------------------------------
def train_vids_step(model, optimizer, g_net, x_full, y_full, num_envs=5, tau=0.001, y_range=(-3,3)):
    model.train()
    optimizer.zero_grad()
    
    env_losses = []
    
    # Pre-compute embeddings
    with torch.no_grad():
        full_embeddings = g_net(x_full)
    
    for l in range(num_envs):
        # A. Create Synthetic Environment (Inverse Bootstrap)
        indices = torch.randperm(len(x_full))
        n_train = 100 # Subsample size
        m_test = 32   # Subsample size
        
        idx_train = indices[:n_train]
        idx_test = indices[n_train:n_train+m_test]
        
        emb_train = full_embeddings[idx_train] 
        emb_test = full_embeddings[idx_test]   
        y_train_real = y_full[idx_train]       
        
        # B. Aggregate Train Embeddings (Deep Sets: Mean)
        emb_train_mean = emb_train.mean(dim=0, keepdim=True)
        emb_train_agg_batch = emb_train_mean.repeat(m_test, 1)
        
        # C. Amortized Inference: Get Theta for each test point
        weight, bias, mu, log_sigma = model(emb_train_agg_batch, emb_test)
        
        # D. Calculate ELBO Terms
        # 1. Likelihood (Reconstruction of TRAINING data)
        pred_train_logits = model.predict_with_theta(emb_train, weight, bias) # (M, N, 1)
        
        # MSE Loss broadcasting
        targets_expanded = y_train_real.unsqueeze(0).expand(m_test, n_train, 1)
        mse = (pred_train_logits - targets_expanded) ** 2
        lik_loss = mse.sum(dim=1).squeeze(-1) # Sum over N -> (M,)

        # 2. Prior Energy
        prior_energy = model.compute_energy_prior(emb_train, emb_test, weight, bias, y_range=y_range)
        
        # 3. Entropy
        entropy = 0.5 * torch.sum(1 + 2 * log_sigma + np.log(2 * np.pi), dim=1)
        
        # Loss = Negative ELBO = Likelihood_Loss - Entropy - Energy
        loss_per_test_point = lik_loss - entropy - prior_energy
        env_losses.append(loss_per_test_point.mean())

    # E. Variance Penalty (OOD Generalization)
    env_losses = torch.stack(env_losses)
    total_loss = env_losses.mean() + tau * torch.var(env_losses)
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()