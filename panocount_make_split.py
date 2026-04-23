import os
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================================
# Dataset path: project's data/dataset by default; override with env PANOCOUNT_DATASET_ROOT if needed
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.environ.get("PANOCOUNT_DATASET_ROOT", os.path.join(PROJECT_ROOT, "data", "dataset"))

IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
JSON_DIR = os.path.join(DATASET_ROOT, "json")
DENSITY_DIR = os.path.join(DATASET_ROOT, "density_maps")

SPLIT_DIR = os.path.join(DATASET_ROOT, "splits")
PREPARED_DIR = os.path.join(DATASET_ROOT, "prepared")

SEED = 42

os.makedirs(SPLIT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(PREPARED_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(PREPARED_DIR, split, "json"), exist_ok=True)
    os.makedirs(os.path.join(PREPARED_DIR, split, "density_maps"), exist_ok=True)


# ==========================================
# BIN FUNCTION
# ==========================================
def get_bin(c):
    if c < 50:
        return "0-49"
    elif c < 100:
        return "50-99"
    elif c < 200:
        return "100-199"
    elif c < 300:
        return "200-299"
    elif c < 500:
        return "300-499"
    else:
        return "500+"


# ==========================================
# READ ALL RECORDS
# ==========================================
records = []
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

for file in sorted(os.listdir(JSON_DIR)):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(JSON_DIR, file)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "human_num" in data:
        count = int(data["human_num"])
    elif "count" in data:
        count = int(data["count"])
    elif "points" in data:
        count = len(data["points"])
    else:
        raise ValueError(f"Could not read count from {json_path}")

    image_id = os.path.splitext(file)[0]

    image_path = None
    for ext in image_exts:
        candidate = os.path.join(IMAGE_DIR, image_id + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        print(f"Warning: image missing for {image_id}, skipping")
        continue

    density_path = os.path.join(DENSITY_DIR, image_id + ".npy")
    if not os.path.exists(density_path):
        density_path = ""

    records.append({
        "id": image_id,
        "count": count,
        "bin": get_bin(count),
        "image_path": image_path,
        "json_path": json_path,
        "density_path": density_path
    })

df = pd.DataFrame(records)

print("Total valid samples:", len(df))
print("\nOverall bin distribution:")
print(df["bin"].value_counts().sort_index())


# ==========================================
# TRUE STRATIFIED SPLIT
# ==========================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["bin"]
)

# Split temp so val ~20% and test ~10% of full data (2/3 of temp -> val, 1/3 -> test)
val_df, test_df = train_test_split(
    temp_df,
    test_size=1/3,
    random_state=SEED,
    stratify=temp_df["bin"]
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
all_df = all_df.sort_values("id").reset_index(drop=True)

print("\nSplit sizes:")
print(all_df["split"].value_counts())


# ==========================================
# SAVE TXT FILES
# ==========================================
for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    txt_path = os.path.join(SPLIT_DIR, f"{split_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for image_id in split_df["id"].tolist():
            f.write(str(image_id) + "\n")


# ==========================================
# SAVE CSV SUMMARIES
# ==========================================
summary = (
    all_df.groupby(["bin", "split"])
    .size()
    .reset_index(name="num_images")
)

summary.to_csv(os.path.join(SPLIT_DIR, "split_bin_summary.csv"), index=False)
all_df.to_csv(os.path.join(SPLIT_DIR, "all_split_records.csv"), index=False)

stats = (
    all_df.groupby("split")["count"]
    .agg(["count", "min", "max", "mean", "median", "std"])
    .round(2)
)
stats.to_csv(os.path.join(SPLIT_DIR, "split_count_stats.csv"))

print("\nBin x Split table:")
pivot = summary.pivot(index="bin", columns="split", values="num_images").fillna(0).astype(int)
pivot = pivot.reindex(["0-49", "50-99", "100-199", "200-299", "300-499", "500+"])
print(pivot)


# ==========================================
# BAR PLOT BY BIN
# ==========================================
ax = pivot.plot(kind="bar", figsize=(10, 6))
ax.set_title("Stratified Train / Val / Test Split by Crowd Count Bin")
ax.set_xlabel("Crowd Count Bin")
ax.set_ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()
barplot_path = os.path.join(SPLIT_DIR, "split_barplot_by_bin.png")
plt.savefig(barplot_path, dpi=200)
plt.close()


# ==========================================
# HISTOGRAM PLOT
# ==========================================
plt.figure(figsize=(10, 6))
plt.hist(train_df["count"], bins=30, alpha=0.5, label="Train")
plt.hist(val_df["count"], bins=30, alpha=0.5, label="Val")
plt.hist(test_df["count"], bins=30, alpha=0.5, label="Test")
plt.title("Crowd Count Distribution Across Splits")
plt.xlabel("Crowd Count")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()
hist_path = os.path.join(SPLIT_DIR, "split_histogram.png")
plt.savefig(hist_path, dpi=200)
plt.close()


# ==========================================
# OPTIONAL: CREATE PREPARED FOLDER WITH COPIES
# ==========================================
COPY_FILES = False   # change to True if you want physical copies

if COPY_FILES:
    for _, row in all_df.iterrows():
        split = row["split"]

        shutil.copy2(
            row["image_path"],
            os.path.join(PREPARED_DIR, split, "images", os.path.basename(row["image_path"]))
        )
        shutil.copy2(
            row["json_path"],
            os.path.join(PREPARED_DIR, split, "json", os.path.basename(row["json_path"]))
        )

        if row["density_path"] != "":
            shutil.copy2(
                row["density_path"],
                os.path.join(PREPARED_DIR, split, "density_maps", os.path.basename(row["density_path"]))
            )

print("\nDone.")
print("Saved split files to:", SPLIT_DIR)
print("Saved bar plot to:", barplot_path)
print("Saved histogram to:", hist_path)