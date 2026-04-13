import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import lightning as L
from unet.unet_parts import *

class UNet(L.LightningModule):
    def __init__(self, n_channels, n_classes, bilinear=False, class_weights=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.class_weights = class_weights

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
        self.outc = OutConv(64, n_classes)

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
        logits = self.outc(x)
        return logits

    def loss(self, pred, target, alpha=1.0, smooth=1e-6):
        # Cross-Entropy
        ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean")
        ce_loss = ce_loss_fn(pred, target)
        # Dice Loss
        pred_soft = torch.softmax(pred, dim=1)
        # print('num classes', pred.shape[1])
        target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
        union = torch.sum(pred_soft + target_one_hot, dim=(2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score.mean()

        return alpha * ce_loss + (1 - alpha) * dice_loss
    
    def dice_coefficient(self, pred, target, smooth=1e-6):
        """Calculate dice coefficient"""
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
        union = torch.sum(pred_soft + target_one_hot, dim=(2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        
        return dice_score.mean()
    
    def create_overlay_image(self, image, mask, alpha=0.5):
        """
        Create an overlay visualization of image with segmentation mask
        
        Args:
            image: tensor of shape (C, H, W) - grayscale or RGB
            mask: tensor of shape (H, W) - segmentation mask with class labels
            alpha: transparency factor for overlay
        
        Returns:
            RGB image tensor with overlay
        """
        # Convert grayscale to RGB if needed
        if image.shape[0] == 1:
            image_rgb = image.repeat(3, 1, 1)
        else:
            image_rgb = image.clone()
        
        # Normalize image to [0, 1] if needed
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0
        
        # Create colored mask (simple colormap: class 0 = background, class 1 = red)
        colored_mask = torch.zeros(3, mask.shape[0], mask.shape[1], device=mask.device)
        colored_mask[0][mask == 1] = 1.0  # Red for class 1
        
        # Blend image and mask
        overlay = (1 - alpha) * image_rgb + alpha * colored_mask
        overlay = torch.clamp(overlay, 0, 1)
        
        return overlay
    

    def log_images(self, batch, pred, stage='train'):
        """Log images with overlays to tensorboard"""
        x, y = batch
        
        # Only log first image in batch to save space
        img = x[0].cpu()
        gt_mask = y[0].cpu()
        pred_mask = torch.argmax(torch.softmax(pred[0], dim=0), dim=0).cpu()
        
        # Create overlays
        gt_overlay = self.create_overlay_image(img, gt_mask)
        pred_overlay = self.create_overlay_image(img, pred_mask)
        
        # Log to tensorboard
        self.logger.experiment.add_image(
            f'{stage}/input_image', 
            img, 
            self.current_epoch
        )
        self.logger.experiment.add_image(
            f'{stage}/ground_truth_overlay', 
            gt_overlay, 
            self.current_epoch
        )
        self.logger.experiment.add_image(
            f'{stage}/prediction_overlay', 
            pred_overlay, 
            self.current_epoch
        )

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        dice = self.dice_coefficient(pred, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log images for first batch of each epoch
        if batch_idx == 0:
            self.log_images(batch, pred, stage='train')
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        dice = self.dice_coefficient(pred, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log images for first batch of each epoch
        if batch_idx == 0:
            self.log_images(batch, pred, stage='val')
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


    def make_prediction(self, img):
        out = self.forward(img)
        out_pred_softmax = torch.softmax(out, dim=1)
        out_pred = torch.argmax(out_pred_softmax, dim=1).squeeze()
        return out_pred



