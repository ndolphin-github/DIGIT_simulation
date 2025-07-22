import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

class DIGITReconstructionDataset(Dataset):
    def __init__(self, root_dir, indenter_list, transform=None):
        self.input_paths = []
        self.target_paths = []
        for name in indenter_list:
            input_files = sorted(glob.glob(os.path.join(root_dir, name, "input_*.npy")))
            target_files = sorted(glob.glob(os.path.join(root_dir, name, "target_*.npy")))
            self.input_paths.extend(input_files)
            self.target_paths.extend(target_files)
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_data = np.load(self.input_paths[idx])  # (H, W, 3)
        target_data = np.load(self.target_paths[idx])  # (H, W, 3)

        # Transpose to (C, H, W)
        input_tensor = torch.tensor(input_data).permute(2, 0, 1).float()
        target_tensor = torch.tensor(target_data).permute(2, 0, 1).float()

        return input_tensor, target_tensor
