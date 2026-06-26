@echo off
setlocal EnableDelayedExpansion

set DATA_ROOT=dataset\new_hotel
set OUT_ROOT=formatted_data\new_hotel
set CKPT_ROOT=checkpoints_new_hotel
set LOG_ROOT=log\new_hotel
set DEVICE=cuda
set EPOCHS=20
set BATCH_SIZE=4
set LR=5e-6
set WEIGHT_DECAY=5e-4
set DROPOUT=0.3

call :run_model text_bert bert-base-uncased bert-base-uncased
if errorlevel 1 exit /b 1

call :run_model text_roberta roberta-base roberta-base
if errorlevel 1 exit /b 1

echo ====================================================================================================
echo === DONE BERT/RoBERTa text-only fine-tune all tasks
echo ====================================================================================================
exit /b 0


:run_model
set MODEL_KEY=%~1
set TEXT_ENCODER=%~2
set TEXT_DIR=%~3

call :ensure_preprocess %MODEL_KEY% %TEXT_ENCODER% %TEXT_DIR%
if errorlevel 1 exit /b 1

for %%T in (mate mabsc macsa macc masc) do (
  call :train_task %MODEL_KEY% %TEXT_ENCODER% %TEXT_DIR% %%T %%T seq
  if errorlevel 1 exit /b 1
)

call :train_task %MODEL_KEY% %TEXT_ENCODER% %TEXT_DIR% quad quad_seq seq
if errorlevel 1 exit /b 1

call :train_task %MODEL_KEY% %TEXT_ENCODER% %TEXT_DIR% quad quad_cls cls
if errorlevel 1 exit /b 1

exit /b 0


:ensure_preprocess
set MODEL_KEY=%~1
set TEXT_ENCODER=%~2
set TEXT_DIR=%~3

if exist "%OUT_ROOT%\%MODEL_KEY%\%TEXT_DIR%\train" if exist "%OUT_ROOT%\%MODEL_KEY%\%TEXT_DIR%\val" if exist "%OUT_ROOT%\%MODEL_KEY%\%TEXT_DIR%\test" (
  echo ====================================================================================================
  echo [SKIP PREPROCESS] %MODEL_KEY% already exists
  echo ====================================================================================================
  exit /b 0
)

echo ====================================================================================================
echo === Preprocess %MODEL_KEY%
echo ====================================================================================================
for %%S in (train val test) do (
  python src\preprocess.py ^
    --set_type %%S ^
    --mode text ^
    --input_path %DATA_ROOT%\%%S.json ^
    --output_root %OUT_ROOT%\%MODEL_KEY% ^
    --text_encoder_name %TEXT_ENCODER% ^
    --max_length 512 ^
    --ignore_empty_span_samples
  if errorlevel 1 exit /b 1
)
exit /b 0


:train_task
set MODEL_KEY=%~1
set TEXT_ENCODER=%~2
set TEXT_DIR=%~3
set TASK_ARG=%~4
set RUN_SUFFIX=%~5
set QUAD_IMPL=%~6
set RUN_NAME=%MODEL_KEY%_%RUN_SUFFIX%
set CKPT_DIR=%CKPT_ROOT%\%RUN_NAME%
set LOG_DIR=%LOG_ROOT%\%RUN_NAME%

if exist "%CKPT_DIR%\best_model.pt" (
  echo ====================================================================================================
  echo [SKIP TRAIN] %RUN_NAME% already has best_model.pt
  echo ====================================================================================================
  exit /b 0
)

echo ====================================================================================================
echo === Fine-tune %MODEL_KEY%: %TASK_ARG%
echo ====================================================================================================
python src\train_text.py ^
  --train_dir %OUT_ROOT%\%MODEL_KEY%\%TEXT_DIR%\train ^
  --val_dir %OUT_ROOT%\%MODEL_KEY%\%TEXT_DIR%\val ^
  --text_model_name %TEXT_ENCODER% ^
  --save_dir %CKPT_DIR% ^
  --log_dir %LOG_DIR% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --lr %LR% ^
  --weight_decay %WEIGHT_DECAY% ^
  --dropout_p %DROPOUT% ^
  --mate_loss_weight 1 ^
  --mote_loss_weight 1 ^
  --macc_loss_weight 1 ^
  --masc_loss_weight 1 ^
  --aope_loss_weight 1 ^
  --task %TASK_ARG% ^
  --quad_impl %QUAD_IMPL% ^
  --device %DEVICE%
exit /b %ERRORLEVEL%
