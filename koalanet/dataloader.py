import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import rawpy


class RawImageDataset(Dataset):
    """Raw Images Dataset"""

    def __init__(self, manifest_csv: str, root_dir: str, transform=None):
        self.manifest: pd.DataFrame = pd.read_csv(manifest_csv)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dark_raw_name = os.path.join(self.root_dir, self.manifest.iloc[idx, 0])
        light_raw_name = os.path.join(self.root_dir, self.manifest.iloc[idx, 1])

        dark_img = rawpy.imread(dark_raw_name).raw_image_visible()
        light_img = rawpy.imread(light_raw_name).raw_image_visible()


