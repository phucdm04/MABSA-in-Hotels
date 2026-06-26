from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from gas_t5_utils import GasExample, build_gas_examples, evaluate_gas_outputs
from metric import format_quad_category_sentiment_report


def load_examples(path: str, task: str) -> List[GasExample]:
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return build_gas_examples(samples, task)


@torch.no_grad()
def generate_predictions(model, tokenizer, examples: List[GasExample], args, device: torch.device) -> List[str]:
    model.eval()
    predictions: List[str] = []
    for start in tqdm(range(0, len(examples), args.batch_size), desc="Testing", leave=False):
        batch_examples = examples[start : start + args.batch_size]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Test text-only GAS-style T5 baseline")
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="log/t5_gas")
    parser.add_argument("--task", type=str, required=True, choices=["quad", "macsa", "macc"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--show_examples", type=int, default=5)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{run_time}_t5_gas_{args.task}_test.log")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_dir).to(device)
    examples = load_examples(args.test_path, args.task)
    predictions = generate_predictions(model, tokenizer, examples, args, device)
    metrics = evaluate_gas_outputs(args.task, examples, predictions)

    log(f"Log file: {log_path}")
    log(f"Device: {args.device}")
    log(f"Checkpoint: {args.ckpt_dir}")
    log(f"Task: {args.task}")
    log(f"Test examples: {len(examples)}")

    if args.task == "macc":
        m = metrics["macc"]
        log(f"test_metric - MACC_MICRO_F1: {m['micro_f1']:.4f} | MACC_MACRO_F1: {m['macro_f1']:.4f}")
        log("=" * 100)
        log("\n[TEST][MACC] classification report")
        log(m["classification_report"])
    elif args.task == "macsa":
        m = metrics["macsa_span"]
        log(
            f"test_metric - MACSA_F1: {m['f1']:.4f} | "
            f"MACSA_PRECISION: {m['precision']:.4f} | MACSA_RECALL: {m['recall']:.4f}"
        )
        log("=" * 100)
        log("\n[TEST][MACSA] generated aspect+category set match")
        log(
            f"precision={m['precision']:.4f} | recall={m['recall']:.4f} | f1={m['f1']:.4f} | "
            f"tp={int(m['tp'])} | fp={int(m['fp'])} | fn={int(m['fn'])} | "
            f"only_fp={int(m['only_fp'])} | only_fn={int(m['only_fn'])} | "
            f"fp_fn={int(m['fp_fn'])} | exact_match={int(m['exact_match'])} | "
            f"num_samples={int(m['num_samples'])}"
        )
    else:
        m = metrics["quad"]
        log(
            f"test_metric - QUAD_F1: {m['f1']:.4f} | "
            f"QUAD_PRECISION: {m['precision']:.4f} | QUAD_RECALL: {m['recall']:.4f}"
        )
        log("=" * 100)
        log("\n[TEST][QuadPrediction] generated exact quad set match")
        log(
            f"precision={m['precision']:.4f} | recall={m['recall']:.4f} | f1={m['f1']:.4f} | "
            f"tp={int(m['tp'])} | fp={int(m['fp'])} | fn={int(m['fn'])}"
        )
        if m.get("category_sentiment") is not None:
            log("=" * 100)
            log("\n[TEST][QuadPrediction] category+sentiment P/R/F1")
            log(format_quad_category_sentiment_report(m["category_sentiment"]))

    if args.show_examples > 0:
        log("=" * 100)
        log("\n[TEST] sample generations")
        for idx, (ex, pred) in enumerate(zip(examples[: args.show_examples], predictions[: args.show_examples]), start=1):
            log(f"[{idx}] json_file={ex.json_file}")
            log(f"source={ex.source}")
            log(f"gold={ex.target}")
            log(f"pred={pred}")


if __name__ == "__main__":
    main()
