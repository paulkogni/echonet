import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import h5py
import json
from torchvision import transforms


class EchonetDynamic(Dataset):

    def __init__(self, root, transform=None):
        
        """
        Args:
            root (str): Path to dataset split
            transform (callable, optional): A function/transform to apply to each image and segmentation
        """

        file_names = os.listdir(root)

        self.examples = [os.path.join(root, file_name) for file_name in file_names]
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
         
def load_data_into_loader(batch_size, path, transform=None):
    """
    Args:
        batch_size (int)
        path (str): Path to the preprocessed dataset
        path_split (str): Path to where the split files are stored
    """
    

    # train_transform = RandomFlipRotate()  # Apply augmentation only for training

    dataset = EchonetDynamic(path)

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    length = len(dataset)
    print("Number of samples:", length)

    return loader