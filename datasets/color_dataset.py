import os
from typing import Tuple, List
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2


def list_images(root_dir: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    # image_rgb: HxWx3 in [0,255], uint8
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV LAB ranges: L in [0, 255], a/b in [0, 255] with 128 as zero
    # Convert to L in [0,1], ab in [-1,1]
    L = lab[:, :, 0:1] / 255.0
    ab = (lab[:, :, 1:3] - 128.0) / 128.0
    return np.concatenate([L, ab], axis=2)


class ColorizationImageDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 256, augment: bool = False):
        super().__init__()
        self.paths = list_images(root_dir)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _augment(self, img: Image.Image) -> Image.Image:
        if self.augment:
            # simple horizontal flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = self._load_image(path)
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        img = self._augment(img)

        img_np = np.array(img)  # HxWx3 uint8
        lab = rgb_to_lab(img_np)  # HxWx3 in L[0,1], ab[-1,1]
        L = lab[:, :, 0:1]
        ab = lab[:, :, 1:3]

        L_t = torch.from_numpy(L.transpose(2, 0, 1)).float()  # 1xHxW
        ab_t = torch.from_numpy(ab.transpose(2, 0, 1)).float()  # 2xHxW

        return {
            "L": L_t,
            "ab": ab_t,
            "path": path,
        }


