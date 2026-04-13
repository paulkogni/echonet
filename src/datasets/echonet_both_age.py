import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import h5py
import json
import pandas as pd
from torchvision import transforms


class EchonetCombined(Dataset):

    def __init__(self, root_dyn, root_ped, filelist_ped_csv, transform=None, min_age_ped=13):
        """
        Args:
            root_dyn (str): Path to preprocessed EchoNet Dynamic dataset
            root_ped (str): Path to preprocessed EchoNet Pediatric dataset
            filelist_ped_csv (str): Path to the FileList.csv for the Pediatric dataset
            transform (callable, optional): A function/transform to apply to each image and segmentation
            min_age_ped (int): Minimum age (inclusive) to include from the Pediatric dataset
        """

        # Load all Dynamic examples
        file_names_dyn = os.listdir(root_dyn)
        self.examples_dyn = [os.path.join(root_dyn, file_name) for file_name in file_names_dyn]

        # Load Pediatric FileList and filter by age
        df_ped = pd.read_csv(filelist_ped_csv)#, delimiter='\t')
        print(df_ped.keys)
        df_ped_filtered = df_ped[df_ped['Age'] >= min_age_ped]

        # Build a set of allowed pediatric filenames (replace .avi with .h5)
        allowed_ped_basenames = set(
            os.path.splitext(fname)[0] + '.h5' for fname in df_ped_filtered['FileName'].values
        )

        # Only include pediatric files that pass the age filter
        file_names_ped = os.listdir(root_ped)
        self.examples_ped = [
            os.path.join(root_ped, file_name)
            for file_name in file_names_ped
            if file_name in allowed_ped_basenames
        ]

        self.examples = self.examples_dyn + self.examples_ped
        self.transform = transform

        print(f"Dynamic samples: {len(self.examples_dyn)}")
        print(f"Pediatric samples (age >= {min_age_ped}): {len(self.examples_ped)}")

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


def load_data_into_loader(batch_size, path_dyn, path_ped, filelist_ped_csv, transform=None, shuffle=True, min_age_ped=13):
    """
    Args:
        batch_size (int)
        path_dyn (str): Path to the preprocessed Dynamic dataset
        path_ped (str): Path to the preprocessed Pediatric dataset
        filelist_ped_csv (str): Path to FileList.csv for the Pediatric dataset
        transform (callable, optional): Transform to apply
        shuffle (bool): Whether to shuffle the data
        min_age_ped (int): Minimum age to include from Pediatric dataset
    """

    dataset = EchonetCombined(path_dyn, path_ped, filelist_ped_csv, transform=transform, min_age_ped=min_age_ped)

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle)

    length = len(dataset)
    print("Number of samples:", length)

    return loader