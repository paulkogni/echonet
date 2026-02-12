import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
import matplotlib.pyplot as plt

from phiseg.torchlayers import Conv2D, Conv2DSequence, ReversibleSequence
import phiseg.utils as utils 

# --- Helper Classes (Modified slightly for Device Compatibility) ---

class DownConvolutionalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, depth=3, padding=True, pool=True, reversible=False):
        super(DownConvolutionalBlock, self).__init__()
        if depth < 1: raise ValueError
        layers = []
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
        if reversible:
            layers.append(ReversibleSequence(input_dim, output_dim, reversible_depth=3))
        else:
            layers.append(Conv2D(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
            if depth > 1:
                for i in range(depth-1):
                    layers.append(Conv2D(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class UpConvolutionalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True, reversible=False):
        super(UpConvolutionalBlock, self).__init__()
        self.bilinear = bilinear
        if self.bilinear:
            if reversible:
                self.upconv_layer = ReversibleSequence(input_dim, output_dim, reversible_depth=2)
            else:
                self.upconv_layer = nn.Sequential(
                    Conv2D(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
                    Conv2D(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                    )
        else:
            raise NotImplementedError

    def forward(self, x, bridge):
        if self.bilinear:
            # FIX: Use size=bridge.shape[2:] instead of scale_factor=2
            # This ensures x is resized to exactly match the skip connection (e.g., 7x7)
            x = nn.functional.interpolate(x, mode='bilinear', size=bridge.shape[2:], align_corners=True)
            x = self.upconv_layer(x)
        
        # These prints verify the fix
        # print('shape 3')
        # print(x.shape[3], bridge.shape[3])
        # print('shape 2')
        # print(x.shape[2], bridge.shape[2])
        
        # Validations
        assert x.shape[3] == bridge.shape[3], f"Width mismatch: x={x.shape[3]}, bridge={bridge.shape[3]}"
        assert x.shape[2] == bridge.shape[2], f"Height mismatch: x={x.shape[2]}, bridge={bridge.shape[2]}"
        
        out = torch.cat([x, bridge], dim=1)
        return out

class SampleZBlock(nn.Module):
    def __init__(self, input_dim, z_dim0=2, depth=2, reversible=False):
        super(SampleZBlock, self).__init__()
        layers = []
        if reversible:
            layers.append(ReversibleSequence(input_dim, input_dim, reversible_depth=3))
        else:
            for i in range(depth):
                layers.append(Conv2D(input_dim, input_dim, kernel_size=3, padding=1))
        self.conv = nn.Sequential(*layers)
        self.mu_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1))
        self.sigma_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1), nn.Softplus())

    def forward(self, pre_z):
        pre_z = self.conv(pre_z)
        mu = self.mu_conv(pre_z)
        sigma = self.sigma_conv(pre_z)
        z = mu + sigma * torch.randn_like(sigma, dtype=torch.float32)
        return mu, sigma, z

class Posterior(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, initializers, padding=True, is_posterior=True, reversible=False):
        super(Posterior, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.latent_levels = 5
        self.resolution_levels = 7
        self.lvl_diff = self.resolution_levels - self.latent_levels
        self.padding = padding
        if is_posterior: self.input_channels += 2
        self.contracting_path = nn.ModuleList()
        
        for i in range(self.resolution_levels):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]
            pool = False if i == 0 else True
            self.contracting_path.append(DownConvolutionalBlock(input, output, initializers, depth=3, padding=padding, pool=pool, reversible=reversible))

        self.upsampling_path = nn.ModuleList()
        for i in reversed(range(self.latent_levels)):
            input = 2
            output = self.num_filters[0]*2
            self.upsampling_path.append(UpConvolutionalBlock(input, output, initializers, padding, reversible=reversible))

        self.sample_z_path = nn.ModuleList()
        for i in reversed(range(self.latent_levels)):
            input = 2*self.num_filters[0] + self.num_filters[i + self.lvl_diff]
            if i == self.latent_levels - 1:
                input = self.num_filters[i + self.lvl_diff]
            self.sample_z_path.append(SampleZBlock(input, depth=2, reversible=reversible))

    def forward(self, patch, segm=None, training_prior=False, z_list=None):
        if segm is not None:
            # Replaced utils call with native torch to ensure device compatibility in Lightning
            with torch.no_grad():
                # One-hot encoding locally to handle device automatically
                segm_one_hot = F.one_hot(segm.long(), num_classes=2).permute(0, 3, 1, 2).float()
                # If segm has channel dim 1, squeeze it first, but F.one_hot expects index tensor
            
            patch = torch.cat([patch, torch.add(segm_one_hot, -0.5)], dim=1)

        blocks = []
        z = [None] * self.latent_levels
        mu = [None] * self.latent_levels
        sigma = [None] * self.latent_levels

        x = patch 
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)
        
        pre_conv = x
        for i, sample_z in enumerate(self.sample_z_path):
            if i != 0:
                pre_conv = self.upsampling_path[i-1](z[-i], blocks[-i])
            mu[-i-1], sigma[-i-1], z[-i-1] = self.sample_z_path[i](pre_conv)
            if training_prior:
                z[-i-1] = z_list[-i-1]
        
        return z, mu, sigma

def increase_resolution(times, input_dim, output_dim):
    module_list = []
    for i in range(times):
        module_list.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True))
        if i != 0: input_dim = output_dim
        module_list.append(Conv2DSequence(input_dim=input_dim, output_dim=output_dim, depth=1))
    return nn.Sequential(*module_list)

class Likelihood(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, latent_levels=5, resolution_levels=7, image_size=(1,112,112), reversible=False, initializers=None, apply_last_layer=True, padding=True):
        super(Likelihood, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.lvl_diff = resolution_levels - latent_levels
        self.image_size = image_size
        self.likelihood_ups_path = nn.ModuleList()
        self.likelihood_post_ups_path = nn.ModuleList()

        for i in reversed(range(self.latent_levels)):
            input = self.num_filters[i]
            if reversible:
                self.likelihood_ups_path.append(ReversibleSequence(input_dim=2, output_dim=input, reversible_depth=2))
            else:
                self.likelihood_ups_path.append(Conv2DSequence(input_dim=2, output_dim=input, depth=2))
            self.likelihood_post_ups_path.append(increase_resolution(times=self.lvl_diff, input_dim=input, output_dim=input))

        self.likelihood_post_c_path = nn.ModuleList()
        for i in range(latent_levels - 1):
            input = self.num_filters[i] + self.num_filters[i + 1 + self.lvl_diff]
            output = self.num_filters[i + self.lvl_diff]
            if reversible:
                self.likelihood_post_c_path.append(ReversibleSequence(input_dim=input, output_dim=output, reversible_depth=2))
            else:
                self.likelihood_post_c_path.append(Conv2DSequence(input_dim=input, output_dim=output, depth=2))

        self.s_layer = nn.ModuleList()
        output = self.num_classes
        for i in reversed(range(self.latent_levels)):
            input = self.num_filters[i + self.lvl_diff]
            self.s_layer.append(Conv2DSequence(input_dim=input, output_dim=output, depth=1, kernel=1, activation=torch.nn.Identity, norm=torch.nn.Identity))

    def forward(self, z):
        s = [None] * self.latent_levels
        post_z = [None] * self.latent_levels
        post_c = [None] * self.latent_levels

        # 1. Expand Latents (z) to Feature Maps (post_z)
        for i in range(self.latent_levels):
            post_z[-i - 1] = self.likelihood_ups_path[i](z[-i - 1])
            post_z[-i - 1] = self.likelihood_post_ups_path[i](post_z[-i - 1])

        # Initialize the deepest level
        post_c[self.latent_levels - 1] = post_z[self.latent_levels - 1]

        # 2. Iteratively Upsample and Concatenate (The Merge Path)
        for i in reversed(range(self.latent_levels - 1)):
            # FIX: Get the exact target size from the level we are merging with
            target_size = post_z[i].shape[2:]
            
            # FIX: Use 'size=target_size' instead of 'scale_factor=2'
            # This forces the 32x32 map to become 28x28 (or whatever fits the current level)
            ups_below = nn.functional.interpolate(
                post_c[i+1], 
                mode='bilinear', 
                size=target_size, 
                align_corners=True
            )
            
            concat = torch.cat([post_z[i], ups_below], dim=1)
            post_c[i] = self.likelihood_post_c_path[i](concat)

        # 3. Generate Segmentation Output
        for i, block in enumerate(self.s_layer):
            s_in = block(post_c[-i-1])
            s[-i-1] = torch.nn.functional.interpolate(
                s_in, 
                size=[self.image_size[1], self.image_size[2]], 
                mode='nearest'
            )
        return s

# --- Main Lightning Module ---

class PHISeg(L.LightningModule):
    def __init__(
        self,
        input_channels, 
        num_classes,
        num_filters,
        latent_levels=5,
        latent_dim=2,
        initializers=None,
        no_convs_fcomb=4,
        beta=10.0,
        image_size=(1,112,112),
        reversible=False,
        apply_last_layer=True,
        exponential_weighting=True,
        padding=True,
        learning_rate=1e-4
    ):
        super(PHISeg, self).__init__()
        self.save_hyperparameters()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_levels = latent_levels
        self.image_size = image_size
        self.exponential_weighting = exponential_weighting
        self.exponential_weight = 4
        
        # Loss weights
        self.kl_divergence_loss_weight = 1.0
        self.residual_multinoulli_loss_weight = 1.0
        self.lr = learning_rate

        self.posterior = Posterior(
            input_channels, num_classes, num_filters, initializers=None, padding=True, reversible=reversible
        )
        self.likelihood = Likelihood(
            input_channels, num_classes, num_filters, initializers=None, apply_last_layer=True, padding=True, image_size=self.image_size, reversible=reversible
        )
        self.prior = Posterior(
            input_channels, num_classes, num_filters, initializers=None, padding=True, is_posterior=False, reversible=reversible
        )

    def accumulate_output(self, output_list, use_softmax=True):
        s_accum = output_list[-1]
        for i in range(len(output_list) - 1):
            s_accum += output_list[i]
        if use_softmax:
            soft_out = torch.nn.functional.softmax(s_accum, dim=1)
            return soft_out
        return s_accum

    def forward(self, x, n_samples=1):
        """
        Inference forward pass: Samples from prior n times and returns the mean prediction.
        """
        # Create a batch of x repeated n_samples times
        bs = x.shape[0]
        xb = x.repeat_interleave(n_samples, dim=0)
        
        # Encode with Prior (unconditioned on mask)
        prior_latent_space, _, _ = self.prior(xb, training_prior=False)
        
        # Decode
        s_out_list = self.likelihood(prior_latent_space)
        accumulated = self.accumulate_output(s_out_list)
        
        # Reshape to [Batch, Samples, C, H, W]
        accumulated = accumulated.view(bs, n_samples, *accumulated.shape[1:])
        
        # Return mean softmax prediction
        return torch.mean(accumulated, dim=1)

    def KL_two_gauss_with_diag_cov(self, mu0, sigma0, mu1, sigma1):
        sigma0_fs = torch.mul(torch.flatten(sigma0, start_dim=1), torch.flatten(sigma0, start_dim=1))
        sigma1_fs = torch.mul(torch.flatten(sigma1, start_dim=1), torch.flatten(sigma1, start_dim=1))
        logsigma0_fs = torch.log(sigma0_fs + 1e-10)
        logsigma1_fs = torch.log(sigma1_fs + 1e-10)
        mu0_f = torch.flatten(mu0, start_dim=1)
        mu1_f = torch.flatten(mu1, start_dim=1)

        return torch.mean(
            0.5*torch.sum(
                torch.div(
                    sigma0_fs + torch.mul((mu1_f - mu0_f), (mu1_f - mu0_f)),
                    sigma1_fs + 1e-10)
                + logsigma1_fs - logsigma0_fs - 1, dim=1)
        )

    def calculate_hierarchical_KL_div_loss(self, posterior_mu, posterior_sigma, prior_mu, prior_sigma):
        kl_loss = 0.
        if self.exponential_weighting:
            level_weights = [self.exponential_weight ** i for i in list(range(self.latent_levels))]
        else:
            level_weights = [1] * self.latent_levels

        for ii, mu_i, sigma_i in zip(reversed(range(self.latent_levels)), reversed(posterior_mu), reversed(posterior_sigma)):
            kl_per_level = self.KL_two_gauss_with_diag_cov(mu_i, sigma_i, prior_mu[ii], prior_sigma[ii])
            kl_loss += self.kl_divergence_loss_weight * level_weights[ii] * kl_per_level
        return kl_loss
    
    def residual_multinoulli_loss(self, reconstruction_list, target):
        criterion = nn.CrossEntropyLoss(reduction='none')
        batch_size = target.shape[0]
        recon_loss = 0.
        s_accumulated = [None] * self.latent_levels

        target_flat = target.view(batch_size, -1).long()

        # Iterate reversed
        for ii in reversed(range(self.latent_levels)):
            s_ii = reconstruction_list[ii]
            
            if ii == self.latent_levels-1:
                s_accumulated[ii] = s_ii
            else:
                s_accumulated[ii] = s_accumulated[ii+1] + s_ii
            
            recon_flat = s_accumulated[ii].view(batch_size, self.num_classes, -1)
            current_loss = torch.mean(torch.sum(criterion(target=target_flat, input=recon_flat), dim=1))
            
            recon_loss += self.residual_multinoulli_loss_weight * current_loss
            
        return recon_loss

    def step(self, batch):
        x, y = batch
        # 1. Posterior Encode (Conditioned on X and Y)
        post_z, post_mu, post_sigma = self.posterior(x, y)
        
        # 2. Prior Encode (Conditioned on X, forced to match Posterior Z during training)
        prior_z, prior_mu, prior_sigma = self.prior(x, training_prior=True, z_list=post_z)
        
        # 3. Decode Posterior Latents
        s_out_list = self.likelihood(post_z)
        
        # 4. Calculate Losses
        kl_loss = self.calculate_hierarchical_KL_div_loss(post_mu, post_sigma, prior_mu, prior_sigma)
        recon_loss = self.residual_multinoulli_loss(s_out_list, y)
        total_loss = recon_loss + kl_loss
        
        # Calculate a final reconstruction for visualization/metrics
        final_recon = self.accumulate_output(s_out_list)
        
        return total_loss, kl_loss, recon_loss, final_recon

    def training_step(self, batch, batch_idx):
        loss, kl_loss, recon_loss, pred_soft = self.step(batch)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True)
        
        # Log images for first batch
        if batch_idx == 0:
            self.log_images(batch, pred_soft, stage='train')
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss, kl_loss, recon_loss, pred_soft = self.step(batch)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        
        # You could implement GED here if desired, but it's expensive
        
        if batch_idx == 0:
            # For validation visualization, we might want to see the "Prior" samples
            # instead of the posterior reconstruction, but step() uses posterior.
            # Let's log the posterior reconstruction for consistency in checking convergence.
            self.log_images(batch, pred_soft, stage='val')
            
        return loss

    def create_overlay_image(self, image, mask, alpha=0.5):
        """Copied from your U-Net example"""
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

    def log_images(self, batch, pred_soft, stage='train'):
        x, y = batch
        img = x[0].cpu()
        gt_mask = y[0].cpu()
        
        # Pred_soft is [B, C, H, W], take argmax
        pred_mask = torch.argmax(pred_soft[0], dim=0).cpu()
        
        gt_overlay = self.create_overlay_image(img, gt_mask)
        pred_overlay = self.create_overlay_image(img, pred_mask)
        
        # Access logger securely
        if self.logger:
            self.logger.experiment.add_image(f'{stage}/input', img, self.current_epoch)
            self.logger.experiment.add_image(f'{stage}/ground_truth', gt_overlay, self.current_epoch)
            self.logger.experiment.add_image(f'{stage}/prediction', pred_overlay, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer