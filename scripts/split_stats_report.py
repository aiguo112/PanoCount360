"""
Report split statistics: per-split counts and mean GT, and per-bin counts.
Reads from data/dataset/splits/*.txt and data/dataset/json for GT counts.
"""
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get("PANOCOUNT_DATASET_ROOT", os.path.join(PROJECT_ROOT, "data", "dataset"))
SPLIT_DIR = os.path.join(DATASET_ROOT, "splits")
JSON_DIR = os.path.join(DATASET_ROOT, "json")


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


BIN_ORDER = ["0-49", "50-99", "100-199", "200-299", "300-499", "500+"]


def load_count(image_id):
    for ext in [".json"]:
        path = os.path.join(JSON_DIR, image_id + ext)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "human_num" in data:
                return int(data["human_num"])
            if "count" in data:
                return int(data["count"])
            if "points" in data:
                return len(data["points"])
    return None


def main():
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(SPLIT_DIR, f"{name}.txt")
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        counts = []
        for i in ids:
            c = load_count(i)
            if c is not None:
                counts.append((i, c))
        splits[name] = counts

    # Per-split stats
    print("=== Per-split statistics ===")
    for name in ["train", "val", "test"]:
        if name not in splits:
            continue
        counts_only = [c for _, c in splits[name]]
        n = len(counts_only)
        if n == 0:
            continue
        mean = sum(counts_only) / n
        variance = sum((x - mean) ** 2 for x in counts_only) / n
        std = variance ** 0.5
        print(f"{name}: n={n}, min={min(counts_only)}, max={max(counts_only)}, mean={mean:.2f}, std={std:.2f}")

    # Per-bin per-split
    print("\n=== Per-bin counts (bin, split, num_images) ===")
    for bin_name in BIN_ORDER:
        for split_name in ["train", "val", "test"]:
            if split_name not in splits:
                continue
            n = sum(1 for _, c in splits[split_name] if get_bin(c) == bin_name)
            if n > 0:
                print(f"{bin_name},{split_name},{n}")

    # Mean GT per split per bin (optional)
    print("\n=== Mean GT count per bin per split ===")
    for bin_name in BIN_ORDER:
        row = []
        for split_name in ["train", "val", "test"]:
            if split_name not in splits:
                row.append("—")
                continue
            vals = [c for _, c in splits[split_name] if get_bin(c) == bin_name]
            if not vals:
                row.append("—")
            else:
                row.append(f"{sum(vals)/len(vals):.1f} (n={len(vals)})")
        print(f"{bin_name}: train={row[0]}, val={row[1]}, test={row[2]}")


if __name__ == "__main__":
    main()
