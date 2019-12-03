import os

import pandas as pd
import numpy as np
import rawpy
import torch
from torch.utils.data import Dataset


def unpack_raw(image: np.ndarray) -> torch.Tensor:
    im = np.expand_dims(image, axis=2)
    im = np.maximum(im - 512, 0) / (16383 - 512)
    im = im * 300
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out: torch.Tensor = torch.from_numpy(
        np.concatenate((im[0:H:2, 0:W:2, :],
                        im[0:H:2, 1:W:2, :],
                        im[1:H:2, 1:W:2, :],
                        im[1:H:2, 0:W:2, :]), axis=2))

    return out.permute([2, 0, 1])


def get_crop(w, h, size):
    x_start = np.random.randint(0, w - size)
    x_end = x_start + size

    y_start = np.random.randint(0, h - size)
    y_end = y_start + size

    return x_start, x_end, y_start, y_end


class RawImageDataset(Dataset):
    """Raw Images Dataset"""

    def __init__(self, manifest_csv: str, root_dir: str, transform=None, crop=512):
        self.manifest: pd.DataFrame = pd.read_csv(manifest_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dark_raw_name = os.path.join(self.root_dir, self.manifest.iloc[idx, 0])
        light_raw_name = os.path.join(self.root_dir, self.manifest.iloc[idx, 1])

        dark_raw = rawpy.imread(dark_raw_name)
        light_raw = rawpy.imread(light_raw_name)

        dark_bayer = dark_raw.raw_image_visible.astype(np.float32)
        light_rgb = torch.from_numpy(light_raw.postprocess()).permute(2, 0, 1)

        if self.crop:
            x0, x1, y0, y1 = get_crop(dark_bayer.shape[0], dark_bayer.shape[1], self.crop)

            dark_bayer = dark_bayer[x0:x1, y0:y1]
            light_rgb = light_rgb[:, x0:x1, y0:y1]

        dark_img = unpack_raw(dark_bayer)
        light_img = light_rgb

        # print(dark_img.shape)
        # print(light_img.shape)

        sample = {'dark': dark_img, 'light': light_img}

        if self.transform:
            sample = self.transform(sample)

        return sample
