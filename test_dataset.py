import os
from torch.utils.data import DataLoader
from data.panocount_dataset import PanoCountDataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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

print("Dataset size:", len(dataset))

sample = dataset[0]
print("Image shape:", sample["image"].shape)
print("Density shape:", sample["density"].shape)
print("Count:", sample["count"].item())
print("ID:", sample["id"])

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

batch = next(iter(loader))
print("Batch image shape:", batch["image"].shape)
print("Batch density shape:", batch["density"].shape)
print("Batch count shape:", batch["count"].shape)
print("Batch ids:", batch["id"])