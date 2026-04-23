"""
After CSRNetPano final training: run evaluation on best_model_FINAL_seed42.pth,
print Test MAE/RMSE, and if MAE in 55-75 save all metrics to results/csrnetpano_final_metrics.csv.
"""
from __future__ import annotations

import os
import sys
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "csrnet_pano")
FINAL_CKPT = os.path.join(CKPT_DIR, "best_model_FINAL_seed42.pth")
RESULTS_JSON = os.path.join(CKPT_DIR, "test_results_FINAL_seed42.json")
METRICS_CSV = os.path.join(PROJECT_ROOT, "results", "csrnetpano_final_metrics.csv")
PYTHON = "C:/Users/Arbi/.conda/envs/seg/python.exe"
ENGINE_DIR = os.path.join(PROJECT_ROOT, "engine")


def compute_metrics(per_image: list) -> dict:
    if not per_image:
        return {}
    pred = np.array([r["pred_count"] for r in per_image], dtype=float)
    gt = np.array([r["gt_count"] for r in per_image], dtype=float)
    abs_err = np.array([r["abs_error"] for r in per_image], dtype=float)
    n = len(abs_err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    denom_mape = np.where(gt > 1e-6, gt, np.nan)
    mape = float(np.nanmean(100.0 * abs_err / denom_mape)) if np.any(np.isfinite(100.0 * abs_err / denom_mape)) else np.nan
    denom_smape = pred + gt
    smape = float(np.mean(200.0 * abs_err / np.where(denom_smape > 1e-6, denom_smape, np.nan))) if np.any(denom_smape > 1e-6) else np.nan
    medae = float(np.median(abs_err))
    mean_gt = float(np.mean(gt))
    nae = mae / mean_gt if mean_gt > 0 else np.nan
    if n > 1 and np.std(pred) > 0 and np.std(gt) > 0:
        r = float(np.corrcoef(pred, gt)[0, 1])
        r2 = r * r
    else:
        r, r2 = np.nan, np.nan
    pct10 = 100.0 * np.mean(abs_err <= 0.10 * gt) if np.all(gt > 0) else 0.0
    pct25 = 100.0 * np.mean(abs_err <= 0.25 * gt) if np.all(gt > 0) else 0.0
    pct50 = 100.0 * np.mean(abs_err <= 0.50 * gt) if np.all(gt > 0) else 0.0
    return {
        "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape,
        "MedAE": medae, "NAE": nae, "Pearson_r": r, "R2": r2,
        "Pct_within_10": pct10, "Pct_within_25": pct25, "Pct_within_50": pct50,
    }


def main():
    if not os.path.isfile(FINAL_CKPT):
        print(f"ERROR: Backup checkpoint not found: {FINAL_CKPT}")
        print("Run training first. Exiting.")
        sys.exit(1)

    # Run evaluation
    cmd = [
        PYTHON,
        os.path.join(PROJECT_ROOT, "engine", "evaluate_model.py"),
        "--model", "csrnet_pano",
        "--checkpoint", FINAL_CKPT,
        "--split", "test",
        "--output", RESULTS_JSON,
    ]
    print("Running evaluation...")
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if rc.returncode != 0:
        print("Evaluation failed. Exiting.")
        sys.exit(rc.returncode)

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_mae = data.get("test_mae")
    test_rmse = data.get("test_rmse")
    if test_mae is None:
        test_mae = float(np.mean([r["abs_error"] for r in data["per_image_results"]]))
    if test_rmse is None:
        pred = np.array([r["pred_count"] for r in data["per_image_results"]])
        gt = np.array([r["gt_count"] for r in data["per_image_results"]])
        test_rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

    print("\n--- Test results ---")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")

    if not (55 <= test_mae <= 75):
        print(f"\nERROR: Test MAE {test_mae:.2f} is outside allowed range 55-75. Stopping.")
        sys.exit(1)

    m = compute_metrics(data["per_image_results"])
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    with open(METRICS_CSV, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in m.items():
            f.write(f"{k},{v}\n")
    print(f"\nMetrics saved to {METRICS_CSV}")


if __name__ == "__main__":
    main()
