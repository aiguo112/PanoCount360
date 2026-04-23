#!/usr/bin/env python
"""
Start training. Runs engine/train_model.py with the correct working directory.

Usage:
  python scripts/train_model.py [--model csrnet_pano] [options...]

Defaults for CSRNetPano (per plan): seed=42, epochs=80, patience=20, weight_decay=1e-4,
5-epoch warmup + cosine LR. Omit --augment-color unless you want color jitter.

Examples:
  # Train CSRNetPano (recommended settings)
  python scripts/train_model.py --model csrnet_pano --epochs 80 --patience 20

  # Train CSRNet baseline
  python scripts/train_model.py --model csrnet

  # Custom batch size / LR
  python scripts/train_model.py --model csrnet_pano --batch-size 4 --lr 5e-6
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "engine", "train_model.py")


def main():
    os.chdir(PROJECT_ROOT)
    cmd = [sys.executable, TRAIN_SCRIPT] + sys.argv[1:]
    if not sys.argv[1:]:
        # No args: run CSRNetPano with plan defaults
        cmd = [
            sys.executable, TRAIN_SCRIPT,
            "--model", "csrnet_pano",
            "--epochs", "80",
            "--patience", "20",
            "--seed", "42",
            "--weight-decay", "1e-4",
        ]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
