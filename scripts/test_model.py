import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from data.panocount_dataset import PanoCountDataset
from torch.utils.data import DataLoader
from models.csrnet import CSRNet
from engine.losses import DensityCountLoss

DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset")
TRAIN_SPLIT = os.path.join(DATASET_ROOT, "splits", "train.txt")

dataset = PanoCountDataset(
    dataset_root=DATASET_ROOT,
    split_file=TRAIN_SPLIT,
    image_size=(512, 1024),
    training=True,
    use_horizontal_roll=True,
    roll_p=0.5
)

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

batch = next(iter(loader))
images = batch["image"]
gt_density = batch["density"]

model = CSRNet(load_pretrained_vgg=True)
criterion = DensityCountLoss(density_weight=1.0, count_weight=0.0)

with torch.no_grad():
    pred_density = model(images)
    out = criterion(pred_density, gt_density)

print("Input image shape:", images.shape)
print("GT density shape:", gt_density.shape)
print("Pred density shape:", pred_density.shape)
print("Loss:", out["loss"].item())
print("Density loss:", out["density_loss"].item())
print("Count loss:", out["count_loss"].item())
print("Pred counts:", pred_density.sum(dim=(1,2,3)))
print("GT counts:", gt_density.sum(dim=(1,2,3)))