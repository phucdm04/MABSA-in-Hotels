from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from gas_t5_utils import GasExample, build_gas_examples, evaluate_gas_outputs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


class GasT5Dataset(Dataset):
    def __init__(self, examples: List[GasExample], tokenizer, max_source_length: int, max_target_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        source = self.tokenizer(
            ex.source,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        target = self.tokenizer(
            text_target=ex.target,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": target["input_ids"],
        }


def collate_gas(batch, tokenizer):
    model_inputs = tokenizer.pad(
        [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch],
        padding=True,
        return_tensors="pt",
    )
    labels = tokenizer.pad(
        [{"input_ids": item["labels"]} for item in batch],
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_examples(path: str, task: str) -> List[GasExample]:
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return build_gas_examples(samples, task)


@torch.no_grad()
def generate_predictions(model, tokenizer, examples: List[GasExample], args, device: torch.device) -> List[str]:
    model.eval()
    predictions: List[str] = []
    for start in tqdm(range(0, len(examples), args.eval_batch_size), desc="Generating", leave=False):
        batch_examples = examples[start : start + args.eval_batch_size]
        encoded = tokenizer(
            [ex.source for ex in batch_examples],
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return predictions


def metric_for_early_stop(task: str, metrics) -> float:
    task = task.lower()
    if task == "macc":
        return float(metrics["macc"]["macro_f1"])
    if task == "macsa":
        return float(metrics["macsa_span"]["f1"])
    return float(metrics["quad"]["f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text-only GAS-style T5 baseline")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="log/t5_gas")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--task", type=str, required=True, choices=["quad", "macsa", "macc"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_if_exists", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_model")
    if args.skip_if_exists and os.path.exists(os.path.join(best_path, "config.json")):
        print(f"[SKIP] checkpoint exists: {best_path}")
        return

    set_seed(args.seed)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{run_time}_t5_gas_{args.task}.log")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    train_examples = load_examples(args.train_path, args.task)
    val_examples = load_examples(args.val_path, args.task)
    train_ds = GasT5Dataset(train_examples, tokenizer, args.max_source_length, args.max_target_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_gas(batch, tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log(f"Log file: {log_path}")
    log(f"Device: {args.device}")
    log(f"Model: {args.model_name}")
    log(f"Task: {args.task}")
    log(f"Train examples: {len(train_examples)} | Val examples: {len(val_examples)}")

    best_score = -1.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
            steps += 1

        val_predictions = generate_predictions(model, tokenizer, val_examples, args, device)
        val_metrics = evaluate_gas_outputs(args.task, val_examples, val_predictions)
        score = metric_for_early_stop(args.task, val_metrics)
        train_loss = running_loss / max(steps, 1)

        log(f"\nEpoch {epoch}/{args.epochs}")
        log(f"train_loss: {train_loss:.6f}")
        if args.task == "macc":
            log(
                f"val_metric - MACC_MICRO_F1: {val_metrics['macc']['micro_f1']:.4f} | "
                f"MACC_MACRO_F1: {val_metrics['macc']['macro_f1']:.4f}"
            )
        elif args.task == "macsa":
            m = val_metrics["macsa_span"]
            log(f"val_metric - MACSA_F1: {m['f1']:.4f} | precision={m['precision']:.4f} | recall={m['recall']:.4f}")
        else:
            m = val_metrics["quad"]
            log(f"val_metric - QUAD_F1: {m['f1']:.4f} | precision={m['precision']:.4f} | recall={m['recall']:.4f}")

        if score > best_score:
            best_score = score
            no_improve = 0
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            log(f"Saved best model to: {best_path}")
        else:
            no_improve += 1
            log(f"No improvement for {no_improve} epoch(s).")
            if no_improve >= args.early_stopping_patience:
                log(f"Early stopping triggered at epoch {epoch}.")
                break


if __name__ == "__main__":
    main()
