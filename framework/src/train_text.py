from __future__ import annotations

import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import ABSABaselineModel
from train import PTDataset, collate_fn, evaluate, train_one_epoch, get_score_summary


def resolve_quad_task(task: str, quad_impl: str) -> str:
    names = {x.strip().lower() for x in task.split(",") if x.strip()}
    quad_aliases = {"quad", "quadprediction", "asqp", "quadra"}
    has_quad = bool(names & quad_aliases)
    names -= quad_aliases
    if has_quad:
        names.add("quad_cls" if quad_impl == "cls" else "quad")
    return ",".join(sorted(names))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text-only ABSA baseline model")
    parser.add_argument("--train_dir", type=str, default="formatted_data/train")
    parser.add_argument("--val_dir", type=str, default="formatted_data/val")
    parser.add_argument("--save_dir", type=str, default="checkpoints_text")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_categories", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--mate_loss_weight", type=float, default=1.0)
    parser.add_argument("--mote_loss_weight", type=float, default=1.0)
    parser.add_argument("--macc_loss_weight", type=float, default=1.0)
    parser.add_argument("--masc_loss_weight", type=float, default=1.0)
    parser.add_argument("--aope_loss_weight", type=float, default=1.0)
    parser.add_argument("--task", type=str, default="mate,mote,macc,masc")
    parser.add_argument("--quad_impl", type=str, default="seq", choices=["seq", "cls"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    effective_task = resolve_quad_task(args.task, args.quad_impl)
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(args.device)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{run_time}_text_{args.epochs}_{args.batch_size}_{args.lr}.log")

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    train_ds = PTDataset(args.train_dir)
    val_ds = PTDataset(args.val_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ABSABaselineModel(
        text_model_name=args.text_model_name,
        num_categories=args.num_categories,
        mate_loss_weight=args.mate_loss_weight,
        mote_loss_weight=args.mote_loss_weight,
        macc_loss_weight=args.macc_loss_weight,
        masc_loss_weight=args.masc_loss_weight,
        aope_loss_weight=args.aope_loss_weight,
        dropout_p=args.dropout_p,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    no_improve_epochs = 0

    log(f"Log file: {log_path}")
    log(f"Device: {args.device}")
    log(f"TASK: {args.task}")
    log(f"QUAD_IMPL: {args.quad_impl}")
    log(f"EFFECTIVE_TASK: {effective_task}")
    log(f"Seed: {args.seed}")
    log(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, log_fn=log, epoch_idx=epoch, task=effective_task)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        train_score = get_score_summary(train_metrics)
        val_score = get_score_summary(val_metrics)
        if "quad_cls" in effective_task:
            mean_score = val_score["quad_cls_f1"]
        elif "quad" in effective_task:
            mean_score = val_score["quad_f1"]
        elif "mabsc" in effective_task:
            mean_score = val_score["mabsc_span_f1"]
        elif "macsa" in effective_task:
            mean_score = val_score["macsa_span_f1"]
        else:
            mean_score = (
                val_score["mate_span_f1"]
                + val_score["mote_span_f1"]
                + val_score["macc_macro_f1"]
                + val_score["masc_macro_f1"]
                + val_score["aope_macro_f1"]
            ) / 5.0

        log(f"\nEpoch {epoch}/{args.epochs}")
        log(f"train_loss: {train_loss:.6f}")
        log(
            f"train_metric - MATE_SPAN_F1: {train_score['mate_span_f1']:.4f} | MOTE_SPAN_F1: {train_score['mote_span_f1']:.4f} | "
            f"MACC_MICRO_F1: {train_score['macc_micro_f1']:.4f} | MACC_MACRO_F1: {train_score['macc_macro_f1']:.4f} | "
            f"MASC_ACCURACY: {train_score['masc_accuracy']:.4f} | MASC_MICRO_F1: {train_score['masc_micro_f1']:.4f} | "
            f"MASC_COUNT: {int(train_score['masc_correct'])}/{int(train_score['masc_total'])} | MASC_MACRO_F1: {train_score['masc_macro_f1']:.4f} | "
            f"AOPE_MICRO_F1: {train_score['aope_micro_f1']:.4f} | AOPE_MACRO_F1: {train_score['aope_macro_f1']:.4f} | "
            f"QUAD_F1: {train_score['quad_f1']:.4f} | QUAD_CLS_F1: {train_score['quad_cls_f1']:.4f}"
        )
        log(
            f"val_metric   - MATE_SPAN_F1: {val_score['mate_span_f1']:.4f} | MOTE_SPAN_F1: {val_score['mote_span_f1']:.4f} | "
            f"MACC_MICRO_F1: {val_score['macc_micro_f1']:.4f} | MACC_MACRO_F1: {val_score['macc_macro_f1']:.4f} | "
            f"MASC_ACCURACY: {val_score['masc_accuracy']:.4f} | MASC_MICRO_F1: {val_score['masc_micro_f1']:.4f} | "
            f"MASC_COUNT: {int(val_score['masc_correct'])}/{int(val_score['masc_total'])} | MASC_MACRO_F1: {val_score['masc_macro_f1']:.4f} | "
            f"AOPE_MICRO_F1: {val_score['aope_micro_f1']:.4f} | AOPE_MACRO_F1: {val_score['aope_macro_f1']:.4f} | "
            f"QUAD_F1: {val_score['quad_f1']:.4f} | QUAD_CLS_F1: {val_score['quad_cls_f1']:.4f}"
        )

        if mean_score > best_val:
            best_val = mean_score
            no_improve_epochs = 0
            ckpt_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            log(f"Saved best model to: {ckpt_path}")
        else:
            no_improve_epochs += 1
            log(f"No improvement for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= args.early_stopping_patience:
                log(f"Early stopping triggered at epoch {epoch}.")
                break



