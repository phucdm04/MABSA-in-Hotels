from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from metric import format_quad_category_sentiment_report
from model import ABSABaselineModel
from test import evaluate_test, masc_class_details, metric_or_zero, task_enabled
from train import PTDataset, collate_fn


def resolve_quad_task(task: str, quad_impl: str) -> str:
    names = {x.strip().lower() for x in task.split(",") if x.strip()}
    quad_aliases = {"quad", "quadprediction", "asqp", "quadra"}
    has_quad = bool(names & quad_aliases)
    names -= quad_aliases
    if has_quad:
        names.add("quad_cls" if quad_impl == "cls" else "quad")
    return ",".join(sorted(names))


def load_compatible_state(model: torch.nn.Module, ckpt_path: str, device: torch.device, log_fn) -> None:
    raw_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = model.state_dict()
    converted_state = {}
    skipped = []

    for key, value in raw_state.items():
        if key not in model_state:
            skipped.append((key, "missing_in_current_model"))
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(model_state[key].shape)}"))
            continue
        converted_state[key] = value

    missing, unexpected = model.load_state_dict(converted_state, strict=False)
    if skipped:
        log_fn(f"Skipped incompatible checkpoint tensors: {len(skipped)}")
        for key, reason in skipped[:20]:
            log_fn(f"  skip {key}: {reason}")
        if len(skipped) > 20:
            log_fn(f"  ... {len(skipped) - 20} more skipped tensor(s)")
    if missing:
        log_fn(f"Missing tensors initialized from current model: {len(missing)}")
    if unexpected:
        log_fn(f"Unexpected tensors ignored: {len(unexpected)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text-only ABSA baseline model")
    parser.add_argument("--test_dir", type=str, default="formatted_data/test")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints_text/best_model.pt")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_categories", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--task", type=str, default="mate,mote,macc,masc")
    parser.add_argument("--quad_impl", type=str, default="seq", choices=["seq", "cls"])
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    effective_task = resolve_quad_task(args.task, args.quad_impl)

    os.makedirs(args.log_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{run_time}_test_text.log")

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = torch.device(args.device)
    test_ds = PTDataset(args.test_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ABSABaselineModel(
        text_model_name=args.text_model_name,
        num_categories=args.num_categories,
    ).to(device)

    log(f"Log file: {log_path}")
    log(f"Device: {args.device}")
    log("Config:")
    log(f"  TEST_DIR={args.test_dir}")
    log(f"  CKPT_PATH={args.ckpt_path}")
    log(f"  BATCH_SIZE={args.batch_size}")
    log(f"  TEXT_MODEL_NAME={args.text_model_name}")
    log(f"  NUM_CATEGORIES={args.num_categories}")
    log(f"  TASK={args.task}")
    log(f"  QUAD_IMPL={args.quad_impl}")
    log(f"  EFFECTIVE_TASK={effective_task}")
    load_compatible_state(model, args.ckpt_path, device, log)
    log(f"Test samples: {len(test_ds)}")

    metrics = evaluate_test(model, test_loader, device)
    summary = []
    if task_enabled(effective_task, "mate"):
        summary.append(f"MATE_SPAN_F1: {metric_or_zero(metrics.get('mate_span'), 'f1'):.4f}")
        summary.append(f"MATE_SPAN_PRECISION: {metric_or_zero(metrics.get('mate_span'), 'precision'):.4f}")
        summary.append(f"MATE_SPAN_RECALL: {metric_or_zero(metrics.get('mate_span'), 'recall'):.4f}")
    if task_enabled(effective_task, "mote"):
        summary.append(f"MOTE_SPAN_F1: {metric_or_zero(metrics.get('mote_span'), 'f1'):.4f}")
        summary.append(f"MOTE_SPAN_PRECISION: {metric_or_zero(metrics.get('mote_span'), 'precision'):.4f}")
        summary.append(f"MOTE_SPAN_RECALL: {metric_or_zero(metrics.get('mote_span'), 'recall'):.4f}")
    if task_enabled(effective_task, "macc"):
        summary.append(f"MACC_MICRO_F1: {metric_or_zero(metrics.get('macc'), 'micro_f1'):.4f}")
        summary.append(f"MACC_MACRO_F1: {metric_or_zero(metrics.get('macc'), 'macro_f1'):.4f}")
    if task_enabled(effective_task, "masc"):
        summary.append(f"MASC_ACCURACY: {metric_or_zero(metrics.get('masc'), 'accuracy'):.4f}")
        summary.append(f"MASC_MICRO_F1: {metric_or_zero(metrics.get('masc'), 'micro_f1'):.4f}")
        summary.append(f"MASC_CORRECT: {int(metric_or_zero(metrics.get('masc'), 'correct'))}")
        summary.append(f"MASC_TOTAL: {int(metric_or_zero(metrics.get('masc'), 'total'))}")
        summary.append(f"MASC_MACRO_F1: {metric_or_zero(metrics.get('masc'), 'macro_f1'):.4f}")
    if task_enabled(effective_task, "aope"):
        summary.append(f"AOPE_F1: {metric_or_zero(metrics.get('aope'), 'f1'):.4f}")
        summary.append(f"AOPE_PRECISION: {metric_or_zero(metrics.get('aope'), 'precision'):.4f}")
        summary.append(f"AOPE_RECALL: {metric_or_zero(metrics.get('aope'), 'recall'):.4f}")
    if task_enabled(effective_task, "mabsc"):
        summary.append(f"MABSC_SPAN_F1: {metric_or_zero(metrics.get('mabsc_span'), 'f1'):.4f}")
        summary.append(f"MABSC_SPAN_PRECISION: {metric_or_zero(metrics.get('mabsc_span'), 'precision'):.4f}")
        summary.append(f"MABSC_SPAN_RECALL: {metric_or_zero(metrics.get('mabsc_span'), 'recall'):.4f}")
    if task_enabled(effective_task, "macsa"):
        summary.append(f"MACSA_SPAN_F1: {metric_or_zero(metrics.get('macsa_span'), 'f1'):.4f}")
        summary.append(f"MACSA_SPAN_PRECISION: {metric_or_zero(metrics.get('macsa_span'), 'precision'):.4f}")
        summary.append(f"MACSA_SPAN_RECALL: {metric_or_zero(metrics.get('macsa_span'), 'recall'):.4f}")
    if task_enabled(effective_task, "quad"):
        summary.append(f"QUAD_F1: {metric_or_zero(metrics.get('quad'), 'f1'):.4f}")
        summary.append(f"QUAD_PRECISION: {metric_or_zero(metrics.get('quad'), 'precision'):.4f}")
        summary.append(f"QUAD_RECALL: {metric_or_zero(metrics.get('quad'), 'recall'):.4f}")
        summary.append(f"QUAD_AS_MABSC_F1: {metric_or_zero(metrics.get('quad_projected', {}).get('mabsc'), 'f1'):.4f}")
        summary.append(f"QUAD_AS_MACSA_F1: {metric_or_zero(metrics.get('quad_projected', {}).get('macsa'), 'f1'):.4f}")
    if task_enabled(effective_task, "quad_cls"):
        summary.append(f"QUAD_CLS_F1: {metric_or_zero(metrics.get('quad_cls'), 'f1'):.4f}")
        summary.append(f"QUAD_CLS_PRECISION: {metric_or_zero(metrics.get('quad_cls'), 'precision'):.4f}")
        summary.append(f"QUAD_CLS_RECALL: {metric_or_zero(metrics.get('quad_cls'), 'recall'):.4f}")
        summary.append(f"QUAD_CLS_AS_MABSC_F1: {metric_or_zero(metrics.get('quad_cls_projected', {}).get('mabsc'), 'f1'):.4f}")
        summary.append(f"QUAD_CLS_AS_MACSA_F1: {metric_or_zero(metrics.get('quad_cls_projected', {}).get('macsa'), 'f1'):.4f}")
    log("test_metric - " + " | ".join(summary))

    if task_enabled(effective_task, "mate") and metrics.get("mate_span") is not None:
        log("=" * 100)
        log("\n[TEST][MATE] predicted aspect span vs gold aspect span")
        log(
            f"precision={metrics['mate_span']['precision']:.4f} | "
            f"recall={metrics['mate_span']['recall']:.4f} | "
            f"f1={metrics['mate_span']['f1']:.4f} | "
            f"tp={int(metrics['mate_span']['tp'])} | "
            f"fp={int(metrics['mate_span']['fp'])} | "
            f"fn={int(metrics['mate_span']['fn'])}"
        )
    if task_enabled(effective_task, "mote") and metrics.get("mote_span") is not None:
        log("=" * 100)
        log("\n[TEST][MOTE] predicted opinion span vs gold opinion span")
        log(
            f"precision={metrics['mote_span']['precision']:.4f} | "
            f"recall={metrics['mote_span']['recall']:.4f} | "
            f"f1={metrics['mote_span']['f1']:.4f} | "
            f"tp={int(metrics['mote_span']['tp'])} | "
            f"fp={int(metrics['mote_span']['fp'])} | "
            f"fn={int(metrics['mote_span']['fn'])}"
        )
    if task_enabled(effective_task, "macc") and metrics.get("macc") is not None:
        log("=" * 100)
        log("\n[TEST][MACC] classification report")
        log(metrics["macc"]["classification_report"])
    if task_enabled(effective_task, "masc") and metrics.get("masc") is not None:
        log("=" * 100)
        log("\n[TEST][MASC] aspect sentiment classification: sentence + aspect -> sentiment")
        log(
            f"accuracy={metrics['masc']['accuracy']:.4f} | "
            f"micro_f1={metrics['masc']['micro_f1']:.4f} | "
            f"correct={int(metrics['masc']['correct'])} | "
            f"total={int(metrics['masc']['total'])}"
        )
        log("\n[TEST][MASC] per-class details")
        log(masc_class_details(metrics["masc"]))
        log(metrics["masc"]["classification_report"])
    if task_enabled(effective_task, "aope") and metrics.get("aope") is not None:
        log("=" * 100)
        log("\n[TEST][MAOPE] relation set match")
        log(
            f"precision={metrics['aope']['precision']:.4f} | recall={metrics['aope']['recall']:.4f} | "
            f"f1={metrics['aope']['f1']:.4f} | tp={int(metrics['aope']['tp'])} | "
            f"fp={int(metrics['aope']['fp'])} | fn={int(metrics['aope']['fn'])}"
        )
    if task_enabled(effective_task, "mabsc") and metrics.get("mabsc_span") is not None:
        log("=" * 100)
        log("\n[TEST][MABSC] predicted span+sentiment vs gold span+sentiment")
        log(
            f"precision={metrics['mabsc_span']['precision']:.4f} | "
            f"recall={metrics['mabsc_span']['recall']:.4f} | "
            f"f1={metrics['mabsc_span']['f1']:.4f} | "
            f"tp={int(metrics['mabsc_span']['tp'])} | "
            f"fp={int(metrics['mabsc_span']['fp'])} | "
            f"fn={int(metrics['mabsc_span']['fn'])} | "
            f"only_fp={int(metrics['mabsc_span']['only_fp'])} | "
            f"only_fn={int(metrics['mabsc_span']['only_fn'])} | "
            f"fp_fn={int(metrics['mabsc_span']['fp_fn'])} | "
            f"exact_match={int(metrics['mabsc_span']['exact_match'])} | "
            f"num_samples={int(metrics['mabsc_span']['num_samples'])}"
        )
    if task_enabled(effective_task, "macsa") and metrics.get("macsa_span") is not None:
        log("=" * 100)
        log("\n[TEST][MACSA] predicted span+category vs gold span+category")
        log(
            f"precision={metrics['macsa_span']['precision']:.4f} | "
            f"recall={metrics['macsa_span']['recall']:.4f} | "
            f"f1={metrics['macsa_span']['f1']:.4f} | "
            f"tp={int(metrics['macsa_span']['tp'])} | "
            f"fp={int(metrics['macsa_span']['fp'])} | "
            f"fn={int(metrics['macsa_span']['fn'])} | "
            f"only_fp={int(metrics['macsa_span']['only_fp'])} | "
            f"only_fn={int(metrics['macsa_span']['only_fn'])} | "
            f"fp_fn={int(metrics['macsa_span']['fp_fn'])} | "
            f"exact_match={int(metrics['macsa_span']['exact_match'])} | "
            f"num_samples={int(metrics['macsa_span']['num_samples'])}"
        )
    if task_enabled(effective_task, "quad") and metrics.get("quad") is not None:
        log("=" * 100)
        log("\n[TEST][QuadPrediction] exact quad set match")
        log(
            f"precision={metrics['quad']['precision']:.4f} | recall={metrics['quad']['recall']:.4f} | "
            f"f1={metrics['quad']['f1']:.4f} | tp={int(metrics['quad']['tp'])} | "
            f"fp={int(metrics['quad']['fp'])} | fn={int(metrics['quad']['fn'])}"
        )
        if metrics["quad"].get("category_sentiment") is not None:
            log("=" * 100)
            log("\n[TEST][QuadPrediction] category+sentiment P/R/F1")
            log(format_quad_category_sentiment_report(metrics["quad"]["category_sentiment"]))
        for label, key in [
            ("MATE", "mate"),
            ("MOTE", "mote"),
            ("MAOPE", "maope"),
            ("MABSC", "mabsc"),
            ("MACSA", "macsa"),
        ]:
            projected = metrics.get("quad_projected", {}).get(key)
            if projected is None:
                continue
            log("=" * 100)
            log(f"\n[TEST][QuadPrediction->{label}] projected from predicted quads")
            log(
                f"precision={projected['precision']:.4f} | "
                f"recall={projected['recall']:.4f} | "
                f"f1={projected['f1']:.4f} | "
                f"tp={int(projected['tp'])} | "
                f"fp={int(projected['fp'])} | "
                f"fn={int(projected['fn'])}"
            )
    if task_enabled(effective_task, "quad_cls") and metrics.get("quad_cls") is not None:
        log("=" * 100)
        log("\n[TEST][QuadPrediction-CLS] exact quad set match from AOPE + MACC + MASC")
        log(
            f"precision={metrics['quad_cls']['precision']:.4f} | recall={metrics['quad_cls']['recall']:.4f} | "
            f"f1={metrics['quad_cls']['f1']:.4f} | tp={int(metrics['quad_cls']['tp'])} | "
            f"fp={int(metrics['quad_cls']['fp'])} | fn={int(metrics['quad_cls']['fn'])}"
        )
        if metrics["quad_cls"].get("category_sentiment") is not None:
            log("=" * 100)
            log("\n[TEST][QuadPrediction-CLS] category+sentiment P/R/F1")
            log(format_quad_category_sentiment_report(metrics["quad_cls"]["category_sentiment"]))
        for label, key in [
            ("MATE", "mate"),
            ("MOTE", "mote"),
            ("MAOPE", "maope"),
            ("MABSC", "mabsc"),
            ("MACSA", "macsa"),
        ]:
            projected = metrics.get("quad_cls_projected", {}).get(key)
            if projected is None:
                continue
            log("=" * 100)
            log(f"\n[TEST][QuadPrediction-CLS->{label}] projected from predicted quads")
            log(
                f"precision={projected['precision']:.4f} | "
                f"recall={projected['recall']:.4f} | "
                f"f1={projected['f1']:.4f} | "
                f"tp={int(projected['tp'])} | "
                f"fp={int(projected['fp'])} | "
                f"fn={int(projected['fn'])}"
            )
