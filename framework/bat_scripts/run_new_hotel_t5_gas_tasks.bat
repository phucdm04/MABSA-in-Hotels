@echo off
setlocal

set DATA_ROOT=dataset\new_hotel
set CKPT_ROOT=checkpoints_new_hotel
set LOG_ROOT=final_log
set MODEL_NAME=t5-base
set MODEL_KEY=t5_gas
set DEVICE=cuda
set EPOCHS=20
set BATCH_SIZE=4
set EVAL_BATCH_SIZE=4
set LR=5e-5
set WEIGHT_DECAY=1e-4
set SEED=2026

echo ====================================================================================================
echo === T5-GAS text-only baseline on new_hotel: Quad, MACSA, MACC
echo ====================================================================================================

call :run_task quad
call :run_task macsa
call :run_task macc

echo ====================================================================================================
echo === Done T5-GAS tasks
echo ====================================================================================================
exit /b 0


:run_task
set TASK=%~1
set RUN_NAME=%MODEL_KEY%_%TASK%
set SAVE_DIR=%CKPT_ROOT%\%RUN_NAME%
set BEST_DIR=%SAVE_DIR%\best_model
set RUN_LOG_DIR=%LOG_ROOT%\%RUN_NAME%

echo ====================================================================================================
echo === Train T5-GAS %TASK%
echo ====================================================================================================
if exist "%BEST_DIR%\config.json" (
  echo [SKIP] checkpoint exists: %BEST_DIR%
) else (
  python src\train_t5_gas.py ^
    --train_path "%DATA_ROOT%\train.json" ^
    --val_path "%DATA_ROOT%\val.json" ^
    --save_dir "%SAVE_DIR%" ^
    --log_dir "%RUN_LOG_DIR%" ^
    --model_name %MODEL_NAME% ^
    --task %TASK% ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --eval_batch_size %EVAL_BATCH_SIZE% ^
    --lr %LR% ^
    --weight_decay %WEIGHT_DECAY% ^
    --early_stopping_patience 1 ^
    --seed %SEED% ^
    --device %DEVICE%
  if errorlevel 1 exit /b 1
)

echo ====================================================================================================
echo === Test T5-GAS %TASK%
echo ====================================================================================================
python src\test_t5_gas.py ^
  --test_path "%DATA_ROOT%\test.json" ^
  --ckpt_dir "%BEST_DIR%" ^
  --log_dir "%RUN_LOG_DIR%" ^
  --task %TASK% ^
  --batch_size %EVAL_BATCH_SIZE% ^
  --device %DEVICE%
if errorlevel 1 exit /b 1
exit /b 0
