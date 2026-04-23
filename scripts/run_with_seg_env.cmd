@echo off
REM Run project scripts using conda env "seg" and project path d:\PanoCrowdCount
set PANOCOUNT_ROOT=d:\PanoCrowdCount
set PANOCOUNT_PYTHON=C:/Users/Arbi/.conda/envs/seg/python.exe

cd /d %PANOCOUNT_ROOT%

if "%~1"=="" (
  echo Usage: run_with_seg_env.cmd [command]
  echo   three_seeds   - run 3-seed validation (train+eval seeds 42,123,777)
  echo   train        - train CSRNetPano seed 42 (80 epochs)
  echo   eval         - evaluate csrnet_pano on test
  echo   eval_csrnet  - evaluate csrnet on test
  goto :eof
)

if "%~1"=="three_seeds" (
  "%PANOCOUNT_PYTHON%" "%PANOCOUNT_ROOT%\scripts\run_three_seeds.py"
  goto :eof
)
if "%~1"=="train" (
  "%PANOCOUNT_PYTHON%" "%PANOCOUNT_ROOT%\engine\train_model.py" --model csrnet_pano --epochs 80 --patience 20 --seed 42 --weight-decay 1e-4
  goto :eof
)
if "%~1"=="eval" (
  "%PANOCOUNT_PYTHON%" "%PANOCOUNT_ROOT%\engine\evaluate_model.py" --model csrnet_pano --split test
  goto :eof
)
if "%~1"=="eval_csrnet" (
  "%PANOCOUNT_PYTHON%" "%PANOCOUNT_ROOT%\engine\evaluate_model.py" --model csrnet --split test
  goto :eof
)

echo Unknown command: %~1
exit /b 1
