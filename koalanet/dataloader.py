import os

import pandas as pd
import numpy as np
import rawpy
import torch
from torch.utils.data import Dataset


def unpack_raw(image: np.ndarray) -> torch.Tensor:
    im = np.expand_dims(image, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out: torch.Tensor = torch.from_numpy(
        np.concatenate((im[0:H:2, 0:W:2, :],
                        im[0:H:2, 1:W:2, :],
                        im[1:H:2, 1:W:2, :],
                        im[1:H:2, 0:W:2, :]), axis=2))

    return out.permute([2, 0, 1])


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

        dark_img = unpack_raw(rawpy.imread(dark_raw_name).raw_image_visible.astype(np.float32))
        light_img = rawpy.imread(light_raw_name).postprocess()

        sample = {'dark': dark_img, 'light': light_img}

        if self.transform:
            sample = self.transform(sample)

        return sample
