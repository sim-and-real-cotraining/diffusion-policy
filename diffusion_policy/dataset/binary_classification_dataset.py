import torch
import numpy as np
from diffusion_policy.dataset.base_dataset import BaseDataset

class BinaryClassificationDataset(BaseDataset):
    def __init__(self, data_0_path, data_1_path):
        # Load data from npy files
        self.data_0 = np.load(data_0_path)
        self.data_1 = np.load(data_1_path)
        self.data_0 = self.data_0.reshape(self.data_0.shape[0], -1)
        self.data_1 = self.data_1.reshape(self.data_1.shape[0], -1)
        
        # Create labels
        self.labels_0 = np.zeros(len(self.data_0))
        self.labels_1 = np.ones(len(self.data_1))
        self.num_zeros = len(self.labels_0)
        self.num_ones = len(self.labels_1)

        # Combine data and labels
        self.data = np.concatenate([self.data_0, self.data_1], axis=0)
        self.labels = np.concatenate([self.labels_0, self.labels_1], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def get_num_ones(self):
        return self.num_ones

    def get_num_zeros(self):
        return self.num_zeros
