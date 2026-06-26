@echo off
setlocal

set DEVICE=cuda
set BATCH_SIZE=4
set IMAGE_ENCODER=google/vit-base-patch16-224
set SIM_TEXT_ENCODER=roberta-base
set GUIDANCE_MODE=cosine
set GUIDANCE_LOSS_WEIGHT=0.5

if not exist final_log mkdir final_log

call :run_sim new_hotel_similarity_roberta_vit_quad_seq formatted_data\new_hotel\similarity_roberta_vit\test checkpoints_new_hotel\similarity_roberta_vit_quad_seq\best_model.pt quad roberta-base seq
call :run_sim new_hotel_similarity_roberta_vit_quad_cls formatted_data\new_hotel\similarity_roberta_vit\test checkpoints_new_hotel\similarity_roberta_vit_quad_cls\best_model.pt quad roberta-base cls
call :run_sim new_hotel_similarity_roberta_vit_mate formatted_data\new_hotel\similarity_roberta_vit\test checkpoints_new_hotel\similarity_roberta_vit_mate\best_model.pt mate roberta-base seq
call :run_sim new_hotel_similarity_roberta_vit_mabsc formatted_data\new_hotel\similarity_roberta_vit\test checkpoints_new_hotel\similarity_roberta_vit_mabsc\best_model.pt mabsc roberta-base seq
call :run_sim new_hotel_similarity_roberta_vit_macsa formatted_data\new_hotel\similarity_roberta_vit\test checkpoints_new_hotel\similarity_roberta_vit_macsa\best_model.pt macsa roberta-base seq
call :run_sim new_hotel_similarity_roberta_vit_masc formatted_data\new_hotel\similarity_roberta_vit\test checkpoints_new_hotel\similarity_roberta_vit_masc\best_model.pt masc roberta-base seq
call :run_sim new_hotel_similarity_roberta_vit_macc formatted_data\new_hotel\similarity_roberta_vit\test checkpoints_new_hotel\similarity_roberta_vit_macc\best_model.pt macc roberta-base seq

echo ====================================================================================================
echo === Done
echo ====================================================================================================
exit /b 0

:run_sim
set RUN_NAME=%~1
set TEST_DIR=%~2
set CKPT_PATH=%~3
set TASK_NAME=%~4
set TEXT_ENCODER=%~5
set QUAD_IMPL=%~6

if not exist "%TEST_DIR%" (
  echo ====================================================================================================
  echo [SKIP] %RUN_NAME% missing test dir: %TEST_DIR%
  echo ====================================================================================================
  exit /b 0
)
if not exist "%CKPT_PATH%" (
  echo ====================================================================================================
  echo [SKIP] %RUN_NAME% missing checkpoint: %CKPT_PATH%
  echo ====================================================================================================
  exit /b 0
)

echo ====================================================================================================
echo [TEST][SIM] %RUN_NAME%
echo ====================================================================================================
python src\test_similarity_guided.py ^
  --test_dir "%TEST_DIR%" ^
  --ckpt_path "%CKPT_PATH%" ^
  --text_model_name "%TEXT_ENCODER%" ^
  --vision_model_name "%IMAGE_ENCODER%" ^
  --guidance_mode %GUIDANCE_MODE% ^
  --guidance_loss_weight %GUIDANCE_LOSS_WEIGHT% ^
  --log_dir "final_log\%RUN_NAME%" ^
  --batch_size %BATCH_SIZE% ^
  --task "%TASK_NAME%" ^
  --quad_impl %QUAD_IMPL% ^
  --device %DEVICE%
if errorlevel 1 exit /b 1
exit /b 0
