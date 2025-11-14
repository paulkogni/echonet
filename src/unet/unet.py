import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from unet.unet_parts import *

class UNet(nn.Module):
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

    def loss(self, pred, target, alpha=0.5, smooth=1e-6):
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


    def make_prediction(self, img):
        out = self.forward(img)
        out_pred_softmax = torch.softmax(out, dim=1)
        out_pred = torch.argmax(out_pred_softmax, dim=1).squeeze()
        return out_pred




def compute_train_loss_and_train(train_loader, model, optimizer, use_gpu, epoch):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        if use_gpu:
            x = x.cuda()
            y = y.cuda()

        output = model(x)
        loss = model.loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss * train_loader.batch_size
        torch.cuda.empty_cache()

    epoch_loss = running_loss / len(train_loader.dataset)


    return epoch_loss


def compute_eval_loss(test_loader, model, use_gpu, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            output = model(x)
            loss = model.loss(output, y)
            running_loss += loss * test_loader.batch_size

        torch.cuda.empty_cache()


    return running_loss / len(test_loader.dataset)


def train_model(
    model,
    train_loader,
    eval_loader,
    optim,
    epochs=1,
    save_model=None,
    save_path=None,
    continue_training_path=None,
    wandb_name=None,
):
    end_epoch = 0
    use_gpu = torch.cuda.is_available()

    if continue_training_path:
        checkpoint = torch.load(continue_training_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if use_gpu:
            model.cuda()
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        end_epoch = checkpoint["epoch"]

    if use_gpu:
        model.cuda()

    best_total_eval_loss = np.inf

    for epoch in range(end_epoch, epochs):
        print("Epoch:", epoch)
        train_loss = compute_train_loss_and_train(train_loader, model, optim, use_gpu, epoch)
        eval_loss = compute_eval_loss(eval_loader, model, use_gpu, epoch)


        if save_model and eval_loss < best_total_eval_loss:
            best_total_eval_loss = eval_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "loss": train_loss,
                },
                f"{save_path}unet_seg_best_eval.pth",
            )
            print("saving best eval model")

        print("training loss:", train_loss)
        print("evaluation loss:", eval_loss)

    return