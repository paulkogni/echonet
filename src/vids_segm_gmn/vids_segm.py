import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
from unet.unet_parts import * 

class UNetFeatureExtractor(nn.Module):
    """
    Acts as g_xi (Fixed Embedding Network).
    Returns feature maps (B, 64, H, W) instead of class logits.
    """
    def __init__(self, n_channels, bilinear=False):
        super(UNetFeatureExtractor, self).__init__()
        self.n_channels = n_channels
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
        # Removed self.outc

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x  # Returns (B, 64, H, W)
    

class SegmentationPredictionHead(nn.Module):
    """
    Bayesian 1x1 Convolution.
    f_theta(g(x))
    """
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        # Params for a 1x1 conv are essentially a Linear layer: (In * Out) + Out (bias)
        self.num_params = embedding_dim * output_dim + output_dim

    def forward(self, feature_map: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C_in, H, W)
            theta: (S, num_params) or (num_params,)
        Returns:
            logits: (S, B, C_out, H, W) or (B, C_out, H, W)
        """
        B, C_in, H, W = feature_map.shape
        
        # 1. Reshape features to behave like a list of pixels: (B*H*W, C_in)
        # Permute to (B, H, W, C_in) then flatten
        x_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, C_in)
        
        # 2. Extract weights from theta
        w_size = self.embedding_dim * self.output_dim
        
        if theta.dim() == 1:
            # Single sample
            Weights = theta[:w_size].view(self.embedding_dim, self.output_dim)
            Bias = theta[w_size:]
            
            # (N_pixels, C_in) @ (C_in, C_out) + Bias -> (N_pixels, C_out)
            out_flat = x_flat @ Weights + Bias
            
            # Reshape back to (B, C_out, H, W)
            return out_flat.view(B, H, W, self.output_dim).permute(0, 3, 1, 2)
            
        else:
            # Multiple samples (S, num_params)
            S = theta.size(0)
            Weights = theta[:, :w_size].view(S, self.embedding_dim, self.output_dim)
            Bias = theta[:, w_size:].unsqueeze(1) # (S, 1, C_out)
            
            # x_flat: (1, N_pixels, C_in)
            x_input = x_flat.unsqueeze(0).expand(S, -1, -1)
            
            # bmm: (S, N_pixels, C_in) @ (S, C_in, C_out) -> (S, N_pixels, C_out)
            out_flat = torch.bmm(x_input, Weights) + Bias
            
            # Reshape back to (S, B, C_out, H, W)
            return out_flat.view(S, B, H, W, self.output_dim).permute(0, 1, 4, 2, 3)
        

class SegmentationPredictionHead(nn.Module):
    """
    Bayesian 1x1 Convolution.
    f_theta(g(x))
    """
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        # Params for a 1x1 conv are essentially a Linear layer: (In * Out) + Out (bias)
        self.num_params = embedding_dim * output_dim + output_dim

    def forward(self, feature_map: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C_in, H, W)
            theta: (S, num_params) or (num_params,)
        Returns:
            logits: (S, B, C_out, H, W) or (B, C_out, H, W)
        """
        B, C_in, H, W = feature_map.shape
        
        # 1. Reshape features to behave like a list of pixels: (B*H*W, C_in)
        # Permute to (B, H, W, C_in) then flatten
        x_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, C_in)
        
        # 2. Extract weights from theta
        w_size = self.embedding_dim * self.output_dim
        
        if theta.dim() == 1:
            # Single sample
            Weights = theta[:w_size].view(self.embedding_dim, self.output_dim)
            Bias = theta[w_size:]
            
            # (N_pixels, C_in) @ (C_in, C_out) + Bias -> (N_pixels, C_out)
            out_flat = x_flat @ Weights + Bias
            
            # Reshape back to (B, C_out, H, W)
            return out_flat.view(B, H, W, self.output_dim).permute(0, 3, 1, 2)
            
        else:
            # Multiple samples (S, num_params)
            S = theta.size(0)
            Weights = theta[:, :w_size].view(S, self.embedding_dim, self.output_dim)
            Bias = theta[:, w_size:].unsqueeze(1) # (S, 1, C_out)
            
            # x_flat: (1, N_pixels, C_in)
            x_input = x_flat.unsqueeze(0).expand(S, -1, -1)
            
            # bmm: (S, N_pixels, C_in) @ (S, C_in, C_out) -> (S, N_pixels, C_out)
            out_flat = torch.bmm(x_input, Weights) + Bias
            
            # Reshape back to (S, B, C_out, H, W)
            return out_flat.view(S, B, H, W, self.output_dim).permute(0, 1, 4, 2, 3)
        


### code for example use
# 1. Load Pre-trained U-Net weights
# unet_model = UNet(n_channels=3, n_classes=2)
# unet_model.load_state_dict(torch.load("unet.pth"))

# # 2. Create VIDS Model
# vids_model = VIDSSegmentation(
#     n_channels=3, 
#     n_classes=2, 
#     embedding_dim=64
# )

# # 3. Transfer weights to VIDS feature extractor
# # We iterate keys because the VIDS extractor doesn't have 'outc'
# pretrained_dict = unet_model.state_dict()
# model_dict = vids_model.embedding_net.state_dict()
# # Filter out unnecessary keys (e.g. outc)
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# vids_model.embedding_net.load_state_dict(model_dict)

# print("Feature extractor weights loaded and frozen.")

# # 4. Train VIDS (Inference Network)
# trainer = L.Trainer(max_epochs=20)
# trainer.fit(vids_model, train_loader, val_loader)