import math
import os

import numpy as np
import pandas as pd
import rawpy
import torch
from torch.utils.data import Dataset


def unpack_raw(image: np.ndarray) -> torch.Tensor:
    im = np.expand_dims(image, axis=2)
    im = np.maximum(im - 512, 0) / (16383 - 512)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out: torch.Tensor = torch.from_numpy(
        np.concatenate((im[0:H:2, 0:W:2, :],
                        im[0:H:2, 1:W:2, :],
                        im[1:H:2, 1:W:2, :],
                        im[1:H:2, 0:W:2, :]), axis=2))

    return out.permute([2, 0, 1])


def get_exposure(img: str) -> float:
    # remove ext
    name = ".".join(img.split('.')[:-1])
    exp = name.split('_')[2][:-1]
    return float(exp)


def get_exposure_ratio(dark_img_name: str, light_img_name: str) -> int:
    return math.ceil(get_exposure(light_img_name) / get_exposure(dark_img_name))


def get_crop(w, h, size):
    x_start = np.random.randint(0, w - size)
    x_end = x_start + size

    y_start = np.random.randint(0, h - size)
    y_end = y_start + size

    return x_start, x_end, y_start, y_end


class RawImageDataset(Dataset):
    """Raw Images Dataset"""

    rgb_light = {}
    bayer_dark = {}

    def __init__(self, manifest_csv: str, root_dir: str, transform=None, crop=512):
        self.manifest: pd.DataFrame = pd.read_csv(manifest_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.crop = crop
        self.rgb_light = {}
        self.rgb_dark = {}
        self.bayer_dark = {}

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(
        #     f"Loading Image {idx} Dark Cache: {len(RawImageDataset.bayer_dark.keys())}
        #     Light Cache: {len(RawImageDataset.rgb_light.keys())}")
        dark_fname = self.manifest.iloc[idx, 0]
        light_fname = self.manifest.iloc[idx, 1]

        if dark_fname in RawImageDataset.bayer_dark:
            dark_bayer = RawImageDataset.bayer_dark[dark_fname]
            # light_rgb = RawImageDataset.rgb_light[light_fname]
            # print(f"CACHE HIT SUCCESS: {idx}/{dark_fname}")
            # dark_rgb = self.rgb_dark[dark_fname]
        else:
            dark_raw_name = os.path.join(self.root_dir, dark_fname)
            with rawpy.imread(dark_raw_name) as dark_raw:
                dark_bayer = dark_raw.raw_image_visible.astype(np.float32)
            RawImageDataset.bayer_dark[dark_fname] = dark_bayer

        if light_fname in RawImageDataset.rgb_light:
            light_rgb = RawImageDataset.rgb_light[light_fname]
            # print(f"CACHE HIT SUCCESS: {idx}/{light_fname}")
        else:
            light_raw_name = os.path.join(self.root_dir, light_fname)
            with rawpy.imread(light_raw_name) as light_raw:
                light_rgb = torch.from_numpy(
                    light_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)).permute(2, 0, 1)
            RawImageDataset.rgb_light[light_fname] = light_rgb

        ratio = get_exposure_ratio(dark_fname, light_fname)

        if self.crop:
            x0, x1, y0, y1 = get_crop(dark_bayer.shape[0], dark_bayer.shape[1], self.crop)

            dark_bayer = dark_bayer[x0:x1, y0:y1]
            light_rgb = light_rgb[:, x0:x1, y0:y1]
            # dark_rgb = dark_rgb[:, x0:x1, y0:y1]

        dark_img = unpack_raw(dark_bayer) * ratio
        light_img = light_rgb

        # print(dark_img.shape)
        # print(light_img.shape)

        sample = {'dark': dark_img, 'light': light_img,
                  # 'dark_rgb': dark_rgb
                  }

        if self.transform:
            sample = self.transform(sample)

        # gc.collect()
        # print(f"Loaded {idx}")

        return sample
