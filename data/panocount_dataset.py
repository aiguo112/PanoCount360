import os
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from data.density_utils import resize_density_map_preserve_count

def read_split_file(split_file: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def find_image_path(images_dir: str, image_id: str) -> str:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    for ext in exts:
        path = os.path.join(images_dir, image_id + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image not found for id={image_id} in {images_dir}")


def resize_density_map_preserve_count(
    density: np.ndarray,
    out_h: int,
    out_w: int
) -> np.ndarray:
    """
    Resize density map while preserving total sum (count).
    """
    in_h, in_w = density.shape[:2]
    if in_h == out_h and in_w == out_w:
        return density.astype(np.float32)

    resized = cv2.resize(density, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    scale_factor = (in_h * in_w) / float(out_h * out_w)
    resized = resized * scale_factor

    return resized.astype(np.float32)


class PanoCountDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split_file: str,
        image_size=(512, 1024),
        training=False,
        use_horizontal_roll=False,
        roll_p=0.5,
        normalize=True
    ):
        super().__init__()

        self.dataset_root = dataset_root
        self.images_dir = os.path.join(dataset_root, "images")
        self.density_dir = os.path.join(dataset_root, "density_maps")

        self.ids = read_split_file(split_file)
        self.image_size = image_size
        self.training = training
        self.use_horizontal_roll = use_horizontal_roll
        self.roll_p = roll_p
        self.normalize = normalize

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, image_id: str) -> np.ndarray:
        image_path = find_image_path(self.images_dir, image_id)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_density(self, image_id: str) -> np.ndarray:
        density_path = os.path.join(self.density_dir, image_id + ".npy")
        if not os.path.exists(density_path):
            raise FileNotFoundError(f"Density map not found: {density_path}")
        density = np.load(density_path).astype(np.float32)
        return density

    def _apply_horizontal_roll(
        self,
        image: np.ndarray,
        density: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ERP-aware augmentation:
        roll image and density horizontally together.
        """
        if not self.training or not self.use_horizontal_roll:
            return image, density

        if np.random.rand() > self.roll_p:
            return image, density

        w = image.shape[1]
        shift = np.random.randint(0, w)

        image = np.roll(image, shift=shift, axis=1)
        density = np.roll(density, shift=shift, axis=1)

        return image, density

    def _resize(
        self,
        image: np.ndarray,
        density: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        out_h, out_w = self.image_size

        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        density = resize_density_map_preserve_count(density, out_h, out_w)

        return image, density

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - self.mean) / self.std
        return image

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        try:
            image = self._load_image(image_id)
            density = self._load_density(image_id)

            image, density = self._apply_horizontal_roll(image, density)
            image, density = self._resize(image, density)

            gt_count = float(density.sum())

            image = self._normalize_image(image)

            image = np.transpose(image, (2, 0, 1))
            density = np.expand_dims(density, axis=0)

            image = torch.from_numpy(image).float()
            density = torch.from_numpy(density).float()
            gt_count = torch.tensor(gt_count, dtype=torch.float32)

            return {
                "image": image,
                "density": density,
                "count": gt_count,
                "id": image_id
            }
        except Exception as e:
            print(f"Dataset error at idx={idx}, id={image_id}: {e}")
            raise