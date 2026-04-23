#!/usr/bin/env python
"""
Run evaluation. Calls engine/evaluate_model.py from project root.

Usage:
  python scripts/evaluate_model.py --model csrnet_pano --split test
  python scripts/evaluate_model.py --model csrnet_pano --split val
  python scripts/evaluate_model.py --model csrnet --split test

With no args: evaluates CSRNetPano on test, then val (both with per-bin MAE).
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "engine", "evaluate_model.py")


def main():
    os.chdir(PROJECT_ROOT)
    if len(sys.argv) > 1:
        cmd = [sys.executable, EVAL_SCRIPT] + sys.argv[1:]
        sys.exit(subprocess.run(cmd).returncode)
    # Default: CSRNetPano on test then val
    for split in ["test", "val"]:
        cmd = [sys.executable, EVAL_SCRIPT, "--model", "csrnet_pano", "--split", split]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
