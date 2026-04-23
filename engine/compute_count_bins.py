import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.panocount_dataset import PanoCountDataset

def main():
    dataset_root = os.path.join(PROJECT_ROOT, "data", "dataset")
    train_split = os.path.join(dataset_root, "splits", "train.txt")

    dataset = PanoCountDataset(
        dataset_root=dataset_root,
        split_file=train_split,
        image_size=(512, 1024),
        training=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    counts = []
    for batch in loader:
        gt_density = batch["density"]
        gt_count = gt_density.sum(dim=(1, 2, 3)).item()
        counts.append(gt_count)

    counts = np.array(counts, dtype=np.float32)

    q1 = np.percentile(counts, 25)
    q2 = np.percentile(counts, 50)
    q3 = np.percentile(counts, 75)

    print("Number of train samples:", len(counts))
    print(f"Min count: {counts.min():.2f}")
    print(f"Max count: {counts.max():.2f}")
    print(f"Q1 (25%): {q1:.2f}")
    print(f"Q2 (50%): {q2:.2f}")
    print(f"Q3 (75%): {q3:.2f}")

if __name__ == "__main__":
    main()