# Fine-Tuning Guide

Assume the project root is:

```powershell
D:\MABSA\framework\baseline
```

Large/runtime folders such as `dataset\`, `checkpoints\`, `log\`, may be empty after cloning.

## 1. Setup Environment

Activate environment:

```powershell
conda activate seminar
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is outside this folder:

```powershell
pip install -r ..\..\requirements.txt
```

## 2. Download Dataset

Download dataset from Google Drive, then place it like this:

```text
dataset\new_hotel\train.json
dataset\new_hotel\val.json
dataset\new_hotel\test.json
dataset\new_hotel_images\
```

If spans are missing, run:

```powershell
python src\temp\add_hmmabsa_spans.py
```

## 3. Fine-Tune SimGui

Run:

```powershell
.\bat_script\run_new_hotel_three_models_tasks_skip.bat
```

This preprocesses data and fine-tunes SimGui on:

```text
MATE
MABSC
MACSA
MACC
MASC
Quad-SEQ
Quad-CLS
```

It skips training if:

```text
checkpoints_new_hotel\<run_name>\best_model.pt
```

already exists.

## 4. Fine-Tune Text-Only BERT/RoBERTa

Run:

```powershell
.\bat_script\run_new_hotel_bert_roberta_finetune_all_tasks.bat
```

This fine-tunes BERT and RoBERTa text-only baselines on the same tasks.

## 5. Fine-Tune T5 GAS Baseline

Run:

```powershell
.\bat_script\run_new_hotel_t5_gas_tasks.bat
```

This fine-tunes T5 on:

```text
Quad
MACSA
MACC
```

## 6. Run Evaluation

Evaluate all main checkpoints:

```powershell
.\bat_script\run_all_checkpoint_tests.bat
```

Evaluate Quad category-sentiment P/R/F1 for BERT, RoBERTa, T5, and SimGui:

```powershell
.\bat_script\run_quad_category_sentiment_four_models_csv.bat
```

Output CSV:

```text
results\quad_category_sentiment_four_models.csv
```

