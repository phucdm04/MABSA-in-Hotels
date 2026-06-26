@echo off
setlocal

set DEVICE=cuda
set BATCH_SIZE=4
set LOG_ROOT=final_log\quad_category_sentiment_four_models
set OUT_CSV=results\quad_category_sentiment_four_models.csv

echo ====================================================================================================
echo === Quad category+sentiment eval: BERT, RoBERTa, T5, SimGui
echo ====================================================================================================

call :eval_text bert bert-base-uncased formatted_data\new_hotel\text_bert\bert-base-uncased\test checkpoints_new_hotel\text_bert_quad_seq\best_model.pt
if errorlevel 1 exit /b 1

call :eval_text roberta roberta-base formatted_data\new_hotel\text_roberta\roberta-base\test checkpoints_new_hotel\text_roberta_quad_seq\best_model.pt
if errorlevel 1 exit /b 1

call :eval_t5 t5 checkpoints_new_hotel\t5_gas_quad\best_model
if errorlevel 1 exit /b 1

call :eval_simgui simgui formatted_data\new_hotel\similarity_roberta_vit\test checkpoints_new_hotel\similarity_roberta_vit_quad_seq\best_model.pt
if errorlevel 1 exit /b 1

echo ====================================================================================================
echo === Collect category+sentiment rows to CSV
echo ====================================================================================================
python src\collect_quad_category_sentiment_csv.py ^
  --log_root "%LOG_ROOT%" ^
  --output_csv "%OUT_CSV%"
if errorlevel 1 exit /b 1

echo ====================================================================================================
echo === Saved CSV: %OUT_CSV%
echo ====================================================================================================
exit /b 0


:eval_text
set MODEL_NAME=%~1
set TEXT_ENCODER=%~2
set TEST_DIR=%~3
set CKPT_PATH=%~4
set RUN_LOG_DIR=%LOG_ROOT%\%MODEL_NAME%

if not exist "%TEST_DIR%" (
  echo [ERROR] Missing %MODEL_NAME% test dir: %TEST_DIR%
  exit /b 1
)
if not exist "%CKPT_PATH%" (
  echo [ERROR] Missing %MODEL_NAME% checkpoint: %CKPT_PATH%
  exit /b 1
)

echo ====================================================================================================
echo === Eval %MODEL_NAME%
echo ====================================================================================================
python src\test_text.py ^
  --test_dir "%TEST_DIR%" ^
  --ckpt_path "%CKPT_PATH%" ^
  --text_model_name %TEXT_ENCODER% ^
  --log_dir "%RUN_LOG_DIR%" ^
  --batch_size %BATCH_SIZE% ^
  --task quad ^
  --quad_impl seq ^
  --device %DEVICE%
exit /b %ERRORLEVEL%


:eval_t5
set MODEL_NAME=%~1
set CKPT_DIR=%~2
set RUN_LOG_DIR=%LOG_ROOT%\%MODEL_NAME%

if not exist "%CKPT_DIR%\config.json" (
  echo [ERROR] Missing %MODEL_NAME% checkpoint dir: %CKPT_DIR%
  exit /b 1
)

echo ====================================================================================================
echo === Eval %MODEL_NAME%
echo ====================================================================================================
python src\test_t5_gas.py ^
  --test_path dataset\new_hotel\test.json ^
  --ckpt_dir "%CKPT_DIR%" ^
  --log_dir "%RUN_LOG_DIR%" ^
  --task quad ^
  --batch_size %BATCH_SIZE% ^
  --device %DEVICE%
exit /b %ERRORLEVEL%


:eval_simgui
set MODEL_NAME=%~1
set TEST_DIR=%~2
set CKPT_PATH=%~3
set RUN_LOG_DIR=%LOG_ROOT%\%MODEL_NAME%

if not exist "%TEST_DIR%" (
  echo [ERROR] Missing %MODEL_NAME% test dir: %TEST_DIR%
  exit /b 1
)
if not exist "%CKPT_PATH%" (
  echo [ERROR] Missing %MODEL_NAME% checkpoint: %CKPT_PATH%
  exit /b 1
)

echo ====================================================================================================
echo === Eval %MODEL_NAME%
echo ====================================================================================================
python src\test_similarity_guided.py ^
  --test_dir "%TEST_DIR%" ^
  --ckpt_path "%CKPT_PATH%" ^
  --text_model_name roberta-base ^
  --vision_model_name google/vit-base-patch16-224 ^
  --guidance_mode cosine ^
  --task quad ^
  --quad_impl seq ^
  --log_dir "%RUN_LOG_DIR%" ^
  --batch_size %BATCH_SIZE% ^
  --device %DEVICE%
exit /b %ERRORLEVEL%
