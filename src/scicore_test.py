import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
# import sys
# sys.path.insert(0, '../src')
import utils
import datasets.echonet_dyn
import unet.unet as unet


path_to_data_train = '/project/home/pfischer95/Documents/echonet_dynamic_preprocessed/TRAIN/'
path_to_data_val = '/project/home/pfischer95/Documents/echonet_dynamic_preprocessed/VAL/'
path_to_data_test = '/project/home/pfischer95/Documents/echonet_dynamic_preprocessed/TEST/'


loader_train = datasets.echonet_dyn.load_data_into_loader(6,path_to_data_train)
loader_val = datasets.echonet_dyn.load_data_into_loader(6,path_to_data_val)
loader_test = datasets.echonet_dyn.load_data_into_loader(6,path_to_data_test)


model = unet.UNet(1,2)

logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = L.Trainer(limit_train_batches=100, max_epochs=10, logger=logger)

trainer.fit(model=model, train_dataloaders=loader_train)