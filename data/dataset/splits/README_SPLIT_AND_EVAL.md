# Split fix and re-evaluation

## What was done

1. **Before split:** Stats saved in `split_stats_before.txt` (70/15/15: train 729, val 156, test 157).
2. **Re-ran** `panocount_make_split.py` (SEED=42) so val ~20%, test ~10%: train 729, val 208, test 105. Stats in `split_count_stats.csv`, `split_bin_summary.csv`, and `split_stats_after.txt`.
3. **Evaluation script** `engine/evaluate_model.py` now supports:
   - `--split val` or `--split test`
   - Per-density-bin MAE/RMSE (same bins as split: 0-49, 50-99, 100-199, 200-299, 300-499, 500+).
   - Writes `val_results.json` or `test_results.json` with `per_bin_mae`.
4. **Training** `engine/train_model.py` now uses:
   - Cosine LR after linear warmup (default 5 epochs), instead of ReduceLROnPlateau.
   - `--warmup-epochs` (default 5).

## Commands to run in your environment (with working torch/CUDA)

### Re-evaluate best checkpoints on new split (val + test, per-bin MAE)

```bash
# CSRNet baseline
python engine/evaluate_model.py --model csrnet --split val
python engine/evaluate_model.py --model csrnet --split test

# CSRNetPano (best checkpoint, e.g. epoch 28)
python engine/evaluate_model.py --model csrnet_pano --split val
python engine/evaluate_model.py --model csrnet_pano --split test
```

Or run all four and print summary:

```bash
python scripts/run_eval_both_splits.py
```

Results are written to `checkpoints/<model>/val_results.json` and `checkpoints/<model>/test_results.json` (test also writes `test_results.json` for backward compatibility). Check whether val–test gap is below 10 MAE points.

### Retrain CSRNetPano with fixed split

```bash
python engine/train_model.py --model csrnet_pano --epochs 80 --patience 20 --seed 42 --weight-decay 1e-4
```

Default: 5-epoch warmup, then cosine decay to 1e-6. No `--augment-color` unless you want it.

### Optional: 3 seeds for mean ± std

After beating baseline, run with seeds 42, 123, 456 and report test MAE mean ± std.
