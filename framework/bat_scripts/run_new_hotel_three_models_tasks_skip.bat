@echo off
setlocal EnableDelayedExpansion

set DATA_ROOT=dataset\new_hotel
set IMAGE_ROOT=dataset\new_hotel_images
set OUT_ROOT=formatted_data\new_hotel
set CKPT_ROOT=checkpoints_new_hotel
set LOG_ROOT=log\new_hotel
set VISION_ENCODER=google/vit-base-patch16-224
set DEVICE=cuda
set EPOCHS=20
set BATCH_SIZE=4
set LR=5e-6
set WEIGHT_DECAY=5e-4
set DROPOUT=0.3
set SEED=2026

call :ensure_preprocess similarity_roberta_vit roberta-base
if errorlevel 1 exit /b 1

for %%T in (mate mabsc macsa macc masc) do (
  call :train_similarity similarity_roberta_vit roberta-base %%T
  if errorlevel 1 exit /b 1
)

call :train_quad_similarity similarity_roberta_vit roberta-base seq
if errorlevel 1 exit /b 1

call :train_quad_similarity similarity_roberta_vit roberta-base cls
if errorlevel 1 exit /b 1

echo ============================================================
echo === DONE new_hotel similarity-guided task run
echo ============================================================
exit /b 0


:ensure_preprocess
set MODEL_KEY=%~1
set TEXT_ENCODER=%~2

if exist "%OUT_ROOT%\%MODEL_KEY%\train" if exist "%OUT_ROOT%\%MODEL_KEY%\val" (
  echo [SKIP PREPROCESS] %MODEL_KEY% already exists
  exit /b 0
)

echo ============================================================
echo === Preprocess %MODEL_KEY%
echo ============================================================
for %%S in (train val test) do (
  python src\preprocess.py ^
    --set_type %%S ^
    --mode both ^
    --input_path %DATA_ROOT%\%%S.json ^
    --output_root %OUT_ROOT%\%MODEL_KEY% ^
    --text_encoder_name %TEXT_ENCODER% ^
    --image_encoder_name %VISION_ENCODER% ^
    --image_root %IMAGE_ROOT% ^
    --max_length 512 ^
    --ignore_empty_span_samples
  if errorlevel 1 exit /b 1
)
exit /b 0


:train_similarity
set MODEL_KEY=%~1
set TEXT_ENCODER=%~2
set TASK_NAME=%~3
set RUN_NAME=%MODEL_KEY%_%TASK_NAME%
set CKPT_DIR=%CKPT_ROOT%\%RUN_NAME%
set LOG_DIR=%LOG_ROOT%\%RUN_NAME%

if exist "%CKPT_DIR%\best_model.pt" (
  echo [SKIP TRAIN] %RUN_NAME% already has best_model.pt
  exit /b 0
)

echo ============================================================
echo === Train similarity-guided %MODEL_KEY%: %TASK_NAME%
echo ============================================================
python src\train_similarity_guided.py ^
  --train_dir %OUT_ROOT%\%MODEL_KEY%\train ^
  --val_dir %OUT_ROOT%\%MODEL_KEY%\val ^
  --text_model_name %TEXT_ENCODER% ^
  --vision_model_name %VISION_ENCODER% ^
  --save_dir %CKPT_DIR% ^
  --log_dir %LOG_DIR% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --lr %LR% ^
  --weight_decay %WEIGHT_DECAY% ^
  --dropout_p %DROPOUT% ^
  --guidance_mode cosine ^
  --guidance_loss_weight 0.5 ^
  --mabsc_loss_weight 1 ^
  --macsa_loss_weight 1 ^
  --macc_loss_weight 1 ^
  --masc_loss_weight 1 ^
  --task %TASK_NAME% ^
  --tensorboard true ^
  --tb_log_interval 100 ^
  --seed %SEED% ^
  --device %DEVICE%
exit /b %ERRORLEVEL%


:train_quad_similarity
set MODEL_KEY=%~1
set TEXT_ENCODER=%~2
set QUAD_IMPL=%~3
set RUN_NAME=%MODEL_KEY%_quad_%QUAD_IMPL%
set CKPT_DIR=%CKPT_ROOT%\%RUN_NAME%
set LOG_DIR=%LOG_ROOT%\%RUN_NAME%

if exist "%CKPT_DIR%\best_model.pt" (
  echo [SKIP TRAIN] %RUN_NAME% already has best_model.pt
  exit /b 0
)

echo ============================================================
echo === Train similarity-guided %MODEL_KEY%: quad %QUAD_IMPL%
echo ============================================================
python src\train_similarity_guided.py ^
  --train_dir %OUT_ROOT%\%MODEL_KEY%\train ^
  --val_dir %OUT_ROOT%\%MODEL_KEY%\val ^
  --text_model_name %TEXT_ENCODER% ^
  --vision_model_name %VISION_ENCODER% ^
  --save_dir %CKPT_DIR% ^
  --log_dir %LOG_DIR% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --lr %LR% ^
  --weight_decay %WEIGHT_DECAY% ^
  --dropout_p %DROPOUT% ^
  --guidance_mode cosine ^
  --guidance_loss_weight 0.5 ^
  --mabsc_loss_weight 1 ^
  --macsa_loss_weight 1 ^
  --macc_loss_weight 1 ^
  --masc_loss_weight 1 ^
  --aope_loss_weight 1 ^
  --task quad ^
  --quad_impl %QUAD_IMPL% ^
  --tensorboard true ^
  --tb_log_interval 100 ^
  --seed %SEED% ^
  --device %DEVICE%
exit /b %ERRORLEVEL%
