import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import h5py
import json
from torchvision import transforms


class EchonetCombined(Dataset):

    def __init__(self, root_dyn, root_ped, transform=None):
        
        """
        Args:
            root (str): Path to dataset split
            transform (callable, optional): A function/transform to apply to each image and segmentation
        """

        file_names_dyn = os.listdir(root_dyn)
        file_names_ped = os.listdir(root_ped)

        self.examples_dyn = [os.path.join(root_dyn, file_name) for file_name in file_names_dyn]
        self.examples_ped = [os.path.join(root_ped, file_name) for file_name in file_names_ped]
        self.examples = self.examples_dyn + self.examples_ped
        self.transform = transform

    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        fname = self.examples[index]

        # load echo image and segmentation
        with h5py.File(fname, 'r') as data:
            ed_es = np.random.choice(['ed', 'es'])
            img = data[ed_es]['image'][()]
            segm = data[ed_es]['mask'][()]


            # convert to torch tensors
            img = torch.Tensor(img).unsqueeze(0)
            segm = torch.Tensor(segm).long()

            if self.transform:
                img, segm = self.transform(img, segm)

            return img, segm
         
def load_data_into_loader(batch_size, path_dyn, path_ped, transform=None, shuffle=True):
    """
    Args:
        batch_size (int)
        path (str): Path to the preprocessed dataset
        path_split (str): Path to where the split files are stored
    """
    

    # train_transform = RandomFlipRotate()  # Apply augmentation only for training

    dataset = EchonetCombined(path_dyn, path_ped, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle)

    length = len(dataset)
    print("Number of samples:", length)

    return loader