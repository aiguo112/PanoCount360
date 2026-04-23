# Three-seed validation and Medium bin improvement

## Implemented

### 1. Three-seed validation (Step 2)

- **Script:** `scripts/run_three_seeds.py`
- **Paths:** Project root `d:\PanoCrowdCount`; Python `C:/Users/Arbi/.conda/envs/seg/python.exe`
- **Option A — CMD helper (recommended):**
  ```cmd
  cd /d d:\PanoCrowdCount
  scripts\run_with_seg_env.cmd three_seeds
  ```
- **Option B — Direct Python:** (from any cwd)
  ```cmd
  set PANOCOUNT_ROOT=d:\PanoCrowdCount
  set PANOCOUNT_PYTHON=C:/Users/Arbi/.conda/envs/seg/python.exe
  %PANOCOUNT_PYTHON% %PANOCOUNT_ROOT%\scripts\run_three_seeds.py
  ```
- **Behaviour:** Trains CSRNetPano for seeds 42, 123, 777 (80 epochs, patience 20, weight-decay 1e-4), evaluates each best checkpoint on the test split, saves each `best_model_seed{seed}.pth`, and writes `scripts/three_seed_results.json` with:
  - Per-seed test MAE and test RMSE
  - Mean ± std of test MAE and test RMSE
  - Flag `all_beats_baseline` (True if all three seeds have test MAE < 78.89)
- **Note:** Training was not run in this environment (PyTorch DLL load error). Run the script locally to get the thesis table.

### 2. Multi-scale density head (Step 3 — Option A)

- **File:** `models/csrnet_pano.py`
- **Change:** Replaced the single `output_layer` (Conv2d 64→1) with:
  - `head_fine`: CircularConv2d(64, 32, 3×3, dilation=1) → ReLU → Conv2d(32,1,1)
  - `head_coarse`: CircularConv2d(64, 32, 3×3, dilation=2) → ReLU → Conv2d(32,1,1)
  - Concat the two density maps (2 ch) → `head_fuse`: Conv2d(2, 1, 1)
  - Then existing `LatitudePrior` and ReLU (unchanged).
- Frontend, backend, and ERP-CBAM are unchanged.

### 3. Retrain and per-bin report (Step 3 continued)

Using conda env **seg** and project `d:\PanoCrowdCount`:

- **Retrain (e.g. seed 42):**
  ```cmd
  scripts\run_with_seg_env.cmd train
  ```
  Or: `C:/Users/Arbi/.conda/envs/seg/python.exe d:\PanoCrowdCount\engine\train_model.py --model csrnet_pano --epochs 80 --patience 20 --seed 42 --weight-decay 1e-4`
- **Evaluate on test:**
  ```cmd
  scripts\run_with_seg_env.cmd eval
  ```
  Or: `C:/Users/Arbi/.conda/envs/seg/python.exe d:\PanoCrowdCount\engine\evaluate_model.py --model csrnet_pano --split test`
- **Per-bin MAE:** Printed by the eval script and stored in `checkpoints/csrnet_pano/test_results.json` under `per_bin_mae`. Use this for the thesis table, especially bins 100–199 and 200–299.

### 4. Val–test gap (Step 4)

- Investigation only; no code or split changes. Findings are in the plan document.

## Deliverables summary

| Deliverable | Status |
|-------------|--------|
| Three-seed runner script | Done (`scripts/run_three_seeds.py`) |
| Three-seed results (mean ± std) | Pending — run script locally |
| Multi-scale head in CSRNetPano | Done (`models/csrnet_pano.py`) |
| Retrain + per-bin MAE report | Pending — run train then eval locally |
