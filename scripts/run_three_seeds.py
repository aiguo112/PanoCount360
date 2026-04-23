"""
Three-seed validation for CSRNetPano (plan Step 2).
Trains with seeds 42, 123, 777; evaluates each on test; reports mean +/- std
for both validation (Val MAE) and test (Test MAE/RMSE). Saves each seed's
best checkpoint as best_model_seed{seed}.pth. If a seed does not converge
(best_val_mae >= 80), a warning is printed but eval still runs (converged: false in JSON).

Training mode (--train-mode):
  subprocess (default): run each seed in a separate Python process. Matches
    manual runs and avoids in-process state (CUDA/cuDNN) affecting later seeds.
  inprocess: run training in the same process (faster startup; may plateau at
    worse Val MAE when running multiple seeds in a row).

Run from project root with conda env that has torch/CUDA, e.g.:
  C:/Users/Arbi/.conda/envs/seg/python.exe d:\PanoCrowdCount\scripts\run_three_seeds.py
  ... or with in-process: ... run_three_seeds.py --train-mode inprocess
"""
import argparse
import os
import sys
import json
import shutil
import subprocess

# Project root from this script's path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Used only for eval subprocess (eval is quick; training is in-process)
PYTHON_EXE = "C:/Users/Arbi/.conda/envs/seg/python.exe"
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "checkpoints", "csrnet_pano", "summary.json")
MAX_VAL_MAE_OK = 80.0

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

SEEDS = [42, 123, 777]
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "csrnet_pano")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "scripts", "three_seed_results.json")


def run_eval(desc):
    """Run evaluate_model.py as subprocess (short run; subprocess is fine)."""
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    eval_script = os.path.join(PROJECT_ROOT, "engine", "evaluate_model.py")
    cmd = f'"{PYTHON_EXE}" "{eval_script}" --model csrnet_pano --split test'
    r = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, env=os.environ.copy())
    if r.returncode != 0:
        print(f"FAILED: {desc} (exit {r.returncode})")
        return False
    return True


def run_train_subprocess(seed):
    """Run training in a separate process (direct Python, same as manual run)."""
    train_script = os.path.join(PROJECT_ROOT, "engine", "train_model.py")
    cmd = [
        PYTHON_EXE,
        train_script,
        "--model", "csrnet_pano",
        "--epochs", "80",
        "--patience", "20",
        "--seed", str(seed),
        "--weight-decay", "1e-4",
    ]
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ.copy())
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Three-seed validation for CSRNetPano")
    parser.add_argument(
        "--train-mode",
        choices=["subprocess", "inprocess"],
        default="subprocess",
        help="subprocess: one process per seed (default, best Val MAE). inprocess: single process.",
    )
    args = parser.parse_args()
    use_subprocess = args.train_mode == "subprocess"

    if use_subprocess:
        train_module = None
    else:
        import engine.train_model as train_module

    results = []
    for seed in SEEDS:
        print(f"\n{'='*60}\nTrain CSRNetPano seed={seed}\n{'='*60}")
        if use_subprocess:
            ok = run_train_subprocess(seed)
            if not ok:
                print(f"FAILED: Train seed={seed} (exit != 0). Skipping eval.")
                continue
        else:
            sys.argv = [
                "train_model.py",
                "--model", "csrnet_pano",
                "--epochs", "80",
                "--patience", "20",
                "--seed", str(seed),
                "--weight-decay", "1e-4",
            ]
            train_module.main()

        # Verify training produced a reasonable model (best_val_mae < 80)
        if not os.path.exists(SUMMARY_JSON):
            raise RuntimeError(
                f"Training completed but {SUMMARY_JSON} not found. "
                f"Cannot verify seed={seed}. Check training log."
            )
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best_val_mae = summary.get("best_val_mae")
        if best_val_mae is None:
            raise RuntimeError(
                f"Seed {seed}: summary.json missing 'best_val_mae'. "
                "Training may have failed. Check training log."
            )
        converged = best_val_mae < MAX_VAL_MAE_OK
        if not converged:
            print(
                f"WARNING: Seed {seed} did not converge (best_val_mae={best_val_mae:.2f} >= {MAX_VAL_MAE_OK}). "
                "Eval will still run; consider re-running this seed manually for better results."
            )
        # Save checkpoint with seed suffix before next run overwrites
        best_src = os.path.join(CKPT_DIR, "best_model.pth")
        best_dst = os.path.join(CKPT_DIR, f"best_model_seed{seed}.pth")
        if os.path.exists(best_src):
            shutil.copy2(best_src, best_dst)
            print(f"Saved {best_dst}")
        # So eval loads this seed's checkpoint: copy seed backup to best_model.pth
        shutil.copy2(best_dst, best_src)
        # Evaluate (subprocess is fine for short eval)
        if not run_eval(f"Eval CSRNetPano seed={seed}"):
            continue
        # Read test_results.json
        tr_path = os.path.join(CKPT_DIR, "test_results.json")
        if os.path.exists(tr_path):
            with open(tr_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mae = data.get("test_mae")
            rmse = data.get("test_rmse")
            results.append({
                "seed": seed,
                "test_mae": mae,
                "test_rmse": rmse,
                "best_val_mae": best_val_mae,
                "converged": converged,
            })
            print(f"Seed {seed}: Val MAE={best_val_mae:.2f} | Test MAE={mae:.2f}, Test RMSE={rmse:.2f}")

    if len(results) < 3:
        print("Could not collect 3 results. Check logs above.")
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": results, "note": "incomplete", "train_mode": args.train_mode}, f, indent=2)
        return

    import numpy as np
    maes = [r["test_mae"] for r in results]
    rmses = [r["test_rmse"] for r in results]
    val_maes = [r["best_val_mae"] for r in results]
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
    val_mae_mean, val_mae_std = np.mean(val_maes), np.std(val_maes)

    seeds_not_converged = [r["seed"] for r in results if not r.get("converged", True)]
    summary = {
        "results": results,
        "val_mae_mean": val_mae_mean,
        "val_mae_std": val_mae_std,
        "test_mae_mean": mae_mean,
        "test_mae_std": mae_std,
        "test_rmse_mean": rmse_mean,
        "test_rmse_std": rmse_std,
        "csrnet_baseline_test_mae": 78.89,
        "all_beats_baseline": all(m < 78.89 for m in maes),
        "convergence_warning": len(seeds_not_converged) > 0,
        "seeds_did_not_converge": seeds_not_converged,
        "train_mode": args.train_mode,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("THREE-SEED SUMMARY (CSRNetPano)")
    print("=" * 60)
    print(f"Train mode: {args.train_mode}")
    print(f"Val MAE:   {val_mae_mean:.2f} +/- {val_mae_std:.2f}")
    print(f"Test MAE:  {mae_mean:.2f} +/- {mae_std:.2f}")
    print(f"Test RMSE: {rmse_mean:.2f} +/- {rmse_std:.2f}")
    print(f"CSRNet baseline (this split) Test MAE: 78.89")
    print(f"All 3 seeds beat baseline: {summary['all_beats_baseline']}")
    if seeds_not_converged:
        print(f"WARNING: Seeds that did not converge (val_mae >= {MAX_VAL_MAE_OK}): {seeds_not_converged}")
    print("=" * 60)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
