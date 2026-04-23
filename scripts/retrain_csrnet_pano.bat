@echo off
REM Retrain CSRNetPano with fixed split: seed=42, epochs=80, patience=20, weight_decay=1e-4
REM Cosine LR with 5-epoch warmup is used by default.
cd /d "%~dp0.."
python engine/train_model.py --model csrnet_pano --epochs 80 --patience 20 --seed 42 --weight-decay 1e-4
pause
