"""
Run evaluation for CSRNet and CSRNetPano on both val and test splits.
Prints overall MAE/RMSE and per-density-bin MAE. Requires checkpoints: csrnet/best_model.pth, csrnet_pano/best_model.pth.
"""
import os
import sys
import subprocess
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

def run_eval(model, split):
    result = subprocess.run(
        [sys.executable, "engine/evaluate_model.py", "--model", model, "--split", split],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=PROJECT_ROOT,
    )
    return result.returncode == 0, result.stdout, result.stderr

def load_results(ckpt_dir, split):
    path = os.path.join(ckpt_dir, f"{split}_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    models = ["csrnet", "csrnet_pano"]
    splits = ["val", "test"]
    for model in models:
        for split in splits:
            ok, out, err = run_eval(model, split)
            if not ok:
                print(f"FAILED {model} {split}:", err or out)
            else:
                print(out)
    # Summary from saved JSONs
    print("\n=== Summary (from saved JSONs) ===")
    for model in models:
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", model)
        for split in splits:
            data = load_results(ckpt_dir, split)
            if data is None:
                print(f"{model} {split}: no results file")
                continue
            mae = data.get(f"{split}_mae", data.get("test_mae"))
            rmse = data.get(f"{split}_rmse", data.get("test_rmse"))
            print(f"{model} {split}: MAE={mae:.2f}, RMSE={rmse:.2f}")
            per_bin = data.get("per_bin_mae", {})
            if per_bin:
                for bin_name, info in per_bin.items():
                    if info.get("n", 0) > 0:
                        print(f"  {bin_name}: n={info['n']}, MAE={info['mae']:.2f}")

if __name__ == "__main__":
    main()
