import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

def log_normal_pdf(x, mu, sigma, eps=1e-8):
    """
    Computes the log probability density of a Normal distribution for each sample,
    summing over the last dimension.
    """
    log2pi = torch.log(torch.tensor(2.0 * np.pi, device=x.device))
    return -0.5 * torch.sum(
        log2pi + 2 * torch.log(sigma + eps) + (x - mu)**2 / (sigma**2 + eps),
        dim=-1
    )

def log_likelihood_binary(logits, y):
    """
    Computes the log likelihood for binary classification.
    logits: shape [batch, 1]
    y: tensor of shape [batch] with values 0 or 1.
    """
    p = torch.sigmoid(logits)
    # Squeeze the last dimension to match y's shape if y is [batch]
    # If y is [batch], p is [batch, 1], we need to squeeze p or unsqueeze y.
    # The original code squeezes axis -1.
    return torch.squeeze(y * torch.log(p + 1e-8) + (1 - y) * torch.log(1 - p + 1e-8), dim=-1)

def log_likelihood_multiclass(logits, y, num_classes):
    """
    Computes the log likelihood for multi-class classification.
    logits: shape [batch, num_classes]
    y: tensor of shape [batch] with integer labels.
    """
    # PyTorch cross_entropy expects class indices (LongTensor)
    # equivalent to -tf.nn.softmax_cross_entropy_with_logits
    # We use reduction='none' to get a value per sample.
    
    # Ensure y is LongTensor and correct shape
    y = y.view(-1).long()
    
    # F.cross_entropy returns positive loss, so we negate it for log likelihood
    return -F.cross_entropy(logits, y, reduction='none')

def log_likelihood_regression(preds, y, sigma=1.0):
    """
    Computes the log likelihood for regression assuming a Gaussian likelihood.
    """
    log_constant = torch.log(torch.tensor(2 * np.pi * sigma**2, device=preds.device))
    return -0.5 * log_constant - 0.5 * ((y - preds) / sigma)**2

def integrated_log_likelihood_classification(logits, binary=True):
    """
    Computes a per-sample integrated log likelihood (negative entropy).
    """
    if binary:
        p = torch.sigmoid(logits)
        p0 = 1 - p
        return torch.squeeze(p * torch.log(p + 1e-8) + p0 * torch.log(p0 + 1e-8), dim=-1)
    else:
        p = torch.softmax(logits, dim=-1)
        return torch.sum(p * torch.log(p + 1e-8), dim=-1)

def integrated_log_likelihood_regression(preds, y_min, y_max, num_mc_samples=10):
    """
    Approximates the integrated log likelihood for regression by Monte Carlo.
    """
    R = num_mc_samples
    # Uniform samples
    y_samples = (y_max - y_min) * torch.rand(R, device=preds.device, dtype=preds.dtype) + y_min
    
    pred_expanded = preds.unsqueeze(0)        # [1, batch, 1]
    y_samples_expanded = y_samples.view(R, 1, 1) # [R, 1, 1]
    
    log2pi = torch.log(torch.tensor(2.0 * np.pi, device=preds.device))
    ll = -0.5 * log2pi - 0.5 * (y_samples_expanded - pred_expanded)**2
    ll = torch.squeeze(ll, dim=-1)
    
    integrated_ll = torch.mean(ll, dim=0)
    return integrated_ll

def normalize_integrated_ll(v):
    """
    Normalizes a vector of integrated log likelihood values.
    Z approx mean(exp(v)).
    """
    Z = torch.mean(torch.exp(v))
    return v - torch.log(Z + 1e-8)

def splits_fn(Xs, Ys, K, m, n):
    """
    Generates splits. 
    Note: Returns numpy arrays. We convert to Tensor inside the loop.
    """
    splits = []
    for k in range(K):
        N = len(Xs)
        fake_test_idx = np.random.choice(N, m, replace=True)
        fake_train_ids = np.random.choice(N, n, replace=True)
        splits.append([Xs[fake_train_ids], Ys[fake_train_ids], Xs[fake_test_idx], Ys[fake_test_idx]])
    return splits

def get_phi(h, gX, gXstar):
    """
    Computes mu and sigma using network h.
    Assumes h takes a concatenated input of [gXstar, gX_mean_tiled].
    """
    # Compute training summary.
    gX_mean = torch.mean(gX, dim=0, keepdim=True)  # [1, d]
    
    # Tile/Expand gX_mean to match gXstar's batch size
    gX_mean_tiled = gX_mean.expand(gXstar.shape[0], -1)  # [n, d]

    # In PyTorch, we typically concat inputs if the model expects multiple features
    # Assuming h is an MLP that takes (d + d) inputs
    inp = torch.cat([gXstar, gX_mean_tiled], dim=-1)
    
    phi = h(inp)
    
    # Split the output of h into mu and sigma (assuming last dim is 2*latent_dim)
    mu, sigma_raw = torch.chunk(phi, 2, dim=-1)
    sigma = F.softplus(sigma_raw) + 1e-6

    return mu, sigma

def train_posterior(Xs, Ys, g, h, optimizer, J, m, n, K=10, tau=1.0, lambda_val=1.0,
                    classification=True, binary=True, num_classes=None,
                    num_mc_samples=10, device='cpu', dtype=torch.float32):
    """
    PyTorch training loop.
    
    Args:
        g, h: PyTorch nn.Modules.
        optimizer: PyTorch optimizer initialized with params of g and h.
        device: 'cpu' or 'cuda'.
    """
    history = []
    
    # Ensure models are on the correct device
    g.to(device)
    h.to(device)

    # Determine y_min/y_max for regression once
    if not classification:
        # Convert all Ys to tensor to find min/max
        Ys_all = torch.tensor(Ys, dtype=dtype, device=device)
        y_min = torch.min(Ys_all)
        y_max = torch.max(Ys_all)
    else:
        y_min, y_max = None, None

    if classification and (not binary):
        if num_classes is None:
            raise ValueError("For multi-class classification, num_classes must be provided.")

    for k in tqdm(range(K)):
        splits = splits_fn(Xs, Ys, J, m, n)
        split_losses = []
        
        # Tracking variables for reporting
        observed_train_ll_last = None
        observed_test_ll_last = None
        train_preds_last = None
        test_preds_last = None
        tmp_Y_last = None
        tmp_Ystar_last = None
        kl_val = None

        optimizer.zero_grad()
        
        # Accumulate losses over J splits
        # Note: In PyTorch, we can accumulate graph operations in a list and stack them
        losses_stack = []

        for j in range(J):
            # 1. Prepare Data
            tmp_X_np, tmp_Y_np, tmp_Xstar_np, tmp_Ystar_np = splits[j]
            
            tmp_X = torch.tensor(tmp_X_np, dtype=dtype, device=device)
            tmp_Xstar = torch.tensor(tmp_Xstar_np, dtype=dtype, device=device)
            
            # Handle Y shape/type
            if classification:
                if binary:
                    # Binary usually expects float for BCE
                    tmp_Y = torch.tensor(tmp_Y_np, dtype=dtype, device=device).unsqueeze(-1)
                    tmp_Ystar = torch.tensor(tmp_Ystar_np, dtype=dtype, device=device).unsqueeze(-1)
                else:
                    # Multiclass usually expects Long/Int for labels, but for calculation we might need shape handling
                    tmp_Y = torch.tensor(tmp_Y_np, device=device)
                    tmp_Ystar = torch.tensor(tmp_Ystar_np, device=device)
            else:
                # Regression
                tmp_Y = torch.tensor(tmp_Y_np, dtype=dtype, device=device)
                tmp_Ystar = torch.tensor(tmp_Ystar_np, dtype=dtype, device=device)
                if len(tmp_Y.shape) == 1:
                    tmp_Y = tmp_Y.unsqueeze(-1)
                    tmp_Ystar = tmp_Ystar.unsqueeze(-1)

            # 2. Compute Embeddings
            gX = g(tmp_X)
            gXstar = g(tmp_Xstar)
            
            # 3. Get Variational Parameters
            mu, sigma = get_phi(h, gX, gXstar)
            
            # 4. Reparameterization
            eps = torch.randn_like(mu)
            theta = mu + sigma * eps
            
            # 5. Posterior Log Prob
            posterior_log_probs = log_normal_pdf(theta, mu, sigma)
            
            # 6. Observed Log Likelihoods
            if classification:
                if binary:
                    theta_avg = torch.mean(theta, dim=0, keepdim=True) # [1, d]
                    
                    # Train (Support)
                    train_logits = torch.matmul(gX, theta_avg.T)
                    obs_train_log_probs_vec = log_likelihood_binary(train_logits, tmp_Y)
                    observed_train_ll = torch.mean(obs_train_log_probs_vec)
                    
                    # Test (Query) - Diagonal interaction
                    # TF: test_logits = diag_part(matmul(gXstar, theta.T))
                    # PyTorch efficient way: sum(A * B, dim=-1)
                    # theta is [m, d], gXstar is [m, d]. We want dot product per sample.
                    test_logits = torch.sum(gXstar * theta, dim=-1) 
                    obs_test_log_probs_vec = log_likelihood_binary(test_logits.unsqueeze(-1), tmp_Ystar)
                    observed_test_ll = torch.mean(obs_test_log_probs_vec)
                    
                    # Integrated LL
                    int_train_ll_vec = integrated_log_likelihood_classification(train_logits, binary=True)
                    int_train_ll_norm = normalize_integrated_ll(int_train_ll_vec)
                    integrated_train_ll = torch.mean(int_train_ll_norm)
                    
                    int_test_ll_vec = integrated_log_likelihood_classification(test_logits.unsqueeze(-1), binary=True)
                    int_test_ll_norm = normalize_integrated_ll(int_test_ll_vec)
                    integrated_test_ll = torch.mean(int_test_ll_norm)
                    
                else:
                    d_rep = gXstar.shape[1]
                    # Reshape theta for multiclass: [batch, d_rep, num_classes]
                    theta_multi = theta.view(theta.shape[0], d_rep, num_classes)
                    
                    theta_avg = torch.mean(theta, dim=0, keepdim=True)
                    theta_avg_multi = theta_avg.view(d_rep, num_classes)
                    
                    # Train
                    train_logits = torch.matmul(gX, theta_avg_multi)
                    obs_train_log_probs_vec = log_likelihood_multiclass(train_logits, tmp_Y, num_classes)
                    observed_train_ll = torch.mean(obs_train_log_probs_vec)
                    
                    # Test
                    # TF: einsum('nd, ndc -> nc')
                    test_logits = torch.einsum('nd, ndc -> nc', gXstar, theta_multi)
                    obs_test_log_probs_vec = log_likelihood_multiclass(test_logits, tmp_Ystar, num_classes)
                    observed_test_ll = torch.mean(obs_test_log_probs_vec)
                    
                    # Integrated
                    int_train_ll_vec = integrated_log_likelihood_classification(train_logits, binary=False)
                    int_train_ll_norm = normalize_integrated_ll(int_train_ll_vec)
                    integrated_train_ll = torch.mean(int_train_ll_norm)
                    
                    int_test_ll_vec = integrated_log_likelihood_classification(test_logits, binary=False)
                    int_test_ll_norm = normalize_integrated_ll(int_test_ll_vec)
                    integrated_test_ll = torch.mean(int_test_ll_norm)

            else:
                # Regression
                theta_avg = torch.mean(theta, dim=0, keepdim=True)
                
                # Train
                train_preds = torch.matmul(gX, theta_avg.T)
                obs_train_log_probs_vec = log_likelihood_regression(train_preds, tmp_Y, sigma=1.0)
                observed_train_ll = torch.mean(obs_train_log_probs_vec)
                
                # Test - dot product per sample
                test_preds = torch.sum(gXstar * theta, dim=-1, keepdim=True)
                obs_test_log_probs_vec = log_likelihood_regression(test_preds, tmp_Ystar, sigma=1.0)
                observed_test_ll = torch.mean(obs_test_log_probs_vec)
                
                # Integrated
                int_train_ll_vec = integrated_log_likelihood_regression(train_preds, y_min, y_max, num_mc_samples)
                int_train_ll_norm = normalize_integrated_ll(int_train_ll_vec)
                integrated_train_ll = torch.mean(int_train_ll_norm)
                
                int_test_ll_vec = integrated_log_likelihood_regression(test_preds, y_min, y_max, num_mc_samples)
                int_test_ll_norm = normalize_integrated_ll(int_test_ll_vec)
                integrated_test_ll = torch.mean(int_test_ll_norm)
                
                train_preds_last = train_preds
                test_preds_last = test_preds
                tmp_Y_last = tmp_Y
                tmp_Ystar_last = tmp_Ystar

            observed_train_ll_last = observed_train_ll
            observed_test_ll_last = observed_test_ll
            
            # 7. Total Loss Calculation
            prior_lp = integrated_train_ll + integrated_test_ll
            posterior_lp = torch.mean(posterior_log_probs)
            kl = posterior_lp - prior_lp
            kl_val = kl # Store for printing

            observed_ll = observed_train_ll + observed_test_ll
            loss_j = -(observed_ll - lambda_val * kl)
            
            losses_stack.append(loss_j)

        # 8. Aggregate Gradient
        losses_tensor = torch.stack(losses_stack)
        split_loss = torch.mean(losses_tensor)
        L_var = torch.var(losses_tensor)
        
        final_loss = split_loss + tau * L_var
        
        final_loss.backward()
        optimizer.step()
        
        history.append(final_loss.item())

        # 9. Printing
        with torch.no_grad():
            if not classification:
                mse_train = torch.mean((train_preds_last - tmp_Y_last)**2)
                mse_test = torch.mean((test_preds_last - tmp_Ystar_last)**2)
                print("--- iteration %d, loss %.4f, split var: %.4f, train MSE: %.4f, test MSE: %.4f, KL: %.4f ---" %
                      (k, final_loss.item(), L_var.item(), mse_train.item(), mse_test.item(), kl_val.item()))
            else:
                train_loss_val = -observed_train_ll_last
                test_loss_val = -observed_test_ll_last
                print("--- iteration %d, loss %.4f, split var: %.4f, train loss: %.4f, test loss: %.4f, KL: %.4f ---" %
                      (k, final_loss.item(), L_var.item(), train_loss_val.item(), test_loss_val.item(), kl_val.item()))

    return h, history