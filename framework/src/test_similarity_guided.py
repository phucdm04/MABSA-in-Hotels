from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model_similarity_guided import SimilarityGuidedMABSAModel
from metric import format_quad_category_sentiment_report
from test import evaluate_test, iou_details_from_confusion, masc_class_details, metric_or_zero, quad_component_details, task_enabled
from train import PTDataset, collate_fn


REPORT_SEPARATOR = "=" * 100


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def resolve_quad_task(task: str, quad_impl: str) -> str:
    names = {x.strip().lower() for x in task.split(",") if x.strip()}
    if not ({"quad", "quadprediction", "asqp", "quadra"} & names):
        return task
    names.discard("quad")
    names.discard("quadprediction")
    names.discard("asqp")
    names.discard("quadra")
    names.discard("quad_cls")
    names.discard("quadclass")
    names.discard("quad_classification")
    if quad_impl == "cls":
        names.add("quad_cls")
    else:
        names.add("quad")
    return ",".join(sorted(names))


def load_compatible_state(model: torch.nn.Module, ckpt_path: str, device: torch.device, log_fn) -> None:
    raw_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = model.state_dict()
    legacy_aope_prefix = "".join(["p", "a", "i", "r"]) + "_head."
    converted_state = {}
    renamed = []
    skipped = []

    for key, value in raw_state.items():
        new_key = key
        if key.startswith(legacy_aope_prefix):
            new_key = "aope_head." + key[len(legacy_aope_prefix) :]
            renamed.append((key, new_key))
        if new_key not in model_state:
            skipped.append((new_key, "missing_in_current_model"))
            continue
        if tuple(model_state[new_key].shape) != tuple(value.shape):
            skipped.append((new_key, f"shape {tuple(value.shape)} != {tuple(model_state[new_key].shape)}"))
            continue
        converted_state[new_key] = value

    missing, unexpected = model.load_state_dict(converted_state, strict=False)
    if renamed:
        log_fn(f"Renamed legacy AOPE checkpoint keys: {len(renamed)}")
    if skipped:
        log_fn(f"Skipped incompatible checkpoint tensors: {len(skipped)}")
        for name, reason in skipped[:20]:
            log_fn(f"  skip {name}: {reason}")
        if len(skipped) > 20:
            log_fn(f"  ... {len(skipped) - 20} more skipped")
    if missing:
        log_fn(f"Missing tensors after compatible load: {len(missing)}")
    if unexpected:
        log_fn(f"Unexpected tensors after compatible load: {len(unexpected)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SimilarityGuidedMABSAModel")
    parser.add_argument("--test_dir", type=str, default="formatted_data/test")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints_similarity_guided/best_model.pt")
    parser.add_argument("--log_dir", type=str, default="log_similarity_guided")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--vision_model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--num_categories", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--guidance_mode", type=str, default="learned", choices=["learned", "cosine"])
    parser.add_argument("--guidance_loss_weight", type=float, default=0.5)
    parser.add_argument("--use_visual_branch", type=str2bool, default=True)
    parser.add_argument("--use_image_cross_attention", type=str2bool, default=True)
    parser.add_argument("--use_visual_gate", type=str2bool, default=True)
    parser.add_argument("--use_visual_guidance", type=str2bool, default=True)
    parser.add_argument("--use_guidance_loss", type=str2bool, default=True)
    parser.add_argument("--mabsc_loss_weight", type=float, default=1.0)
    parser.add_argument("--macsa_loss_weight", type=float, default=1.0)
    parser.add_argument("--maope_impl", type=str, default="ce", choices=["ce", "contrastive"])
    parser.add_argument("--maope_contrastive_weight", type=float, default=0.2)
    parser.add_argument("--task", type=str, default="mate,mote,macc,masc")
    parser.add_argument("--quad_impl", type=str, default="seq", choices=["seq", "cls"])
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    effective_task = resolve_quad_task(args.task, args.quad_impl)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_similarity_guided_test.log")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = torch.device(args.device)
    test_ds = PTDataset(args.test_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = SimilarityGuidedMABSAModel(
        text_model_name=args.text_model_name,
        vision_model_name=args.vision_model_name,
        num_categories=args.num_categories,
        guidance_mode=args.guidance_mode,
        guidance_loss_weight=args.guidance_loss_weight,
        use_visual_branch=args.use_visual_branch,
        use_image_cross_attention=args.use_image_cross_attention,
        use_visual_gate=args.use_visual_gate,
        use_visual_guidance=args.use_visual_guidance,
        use_guidance_loss=args.use_guidance_loss,
        mabsc_loss_weight=args.mabsc_loss_weight,
        macsa_loss_weight=args.macsa_loss_weight,
        maope_impl=args.maope_impl,
        maope_contrastive_weight=args.maope_contrastive_weight,
    ).to(device)
    log(f"Log file: {log_path}")
    log(f"Device: {args.device}")
    log(f"TASK: {args.task}")
    log(f"QUAD_IMPL: {args.quad_impl}")
    log(f"MAOPE_IMPL: {args.maope_impl}")
    log(f"MAOPE_CONTRASTIVE_WEIGHT: {args.maope_contrastive_weight}")
    log(f"USE_VISUAL_BRANCH: {args.use_visual_branch}")
    log(f"USE_IMAGE_CROSS_ATTENTION: {args.use_image_cross_attention}")
    log(f"USE_VISUAL_GATE: {args.use_visual_gate}")
    log(f"USE_VISUAL_GUIDANCE: {args.use_visual_guidance}")
    log(f"USE_GUIDANCE_LOSS: {args.use_guidance_loss}")
    log(f"EFFECTIVE_TASK: {effective_task}")
    load_compatible_state(model, args.ckpt_path, device, log)
    log(f"Test samples: {len(test_ds)}")
    derive_from_mabsc = (
        task_enabled(effective_task, "mabsc")
        and (task_enabled(effective_task, "mate") or task_enabled(effective_task, "masc"))
    )
    metrics = evaluate_test(
        model,
        test_loader,
        device,
        derive_mate_masc_from_mabsc=derive_from_mabsc,
    )

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
        summary.append(f"MACC_MICRO_F1: {metric_or_zero(metrics['macc'], 'micro_f1'):.4f}")
        summary.append(f"MACC_MACRO_F1: {metric_or_zero(metrics['macc'], 'macro_f1'):.4f}")
    if task_enabled(effective_task, "masc"):
        summary.append(f"MASC_ACCURACY: {metric_or_zero(metrics['masc'], 'accuracy'):.4f}")
        summary.append(f"MASC_MICRO_F1: {metric_or_zero(metrics['masc'], 'micro_f1'):.4f}")
        summary.append(f"MASC_CORRECT: {int(metric_or_zero(metrics['masc'], 'correct'))}")
        summary.append(f"MASC_TOTAL: {int(metric_or_zero(metrics['masc'], 'total'))}")
        summary.append(f"MASC_MACRO_F1: {metric_or_zero(metrics['masc'], 'macro_f1'):.4f}")
    if task_enabled(effective_task, "mabsc"):
        summary.append(f"MABSC_SPAN_F1: {metric_or_zero(metrics.get('mabsc_span'), 'f1'):.4f}")
        summary.append(f"MABSC_SPAN_PRECISION: {metric_or_zero(metrics.get('mabsc_span'), 'precision'):.4f}")
        summary.append(f"MABSC_SPAN_RECALL: {metric_or_zero(metrics.get('mabsc_span'), 'recall'):.4f}")
        summary.append(f"MABSC_AS_MATE_F1: {metric_or_zero(metrics.get('mate_from_mabsc'), 'f1'):.4f}")
    if task_enabled(effective_task, "macsa"):
        summary.append(f"MACSA_SPAN_F1: {metric_or_zero(metrics.get('macsa_span'), 'f1'):.4f}")
        summary.append(f"MACSA_SPAN_PRECISION: {metric_or_zero(metrics.get('macsa_span'), 'precision'):.4f}")
        summary.append(f"MACSA_SPAN_RECALL: {metric_or_zero(metrics.get('macsa_span'), 'recall'):.4f}")
        summary.append(f"MACSA_AS_MATE_F1: {metric_or_zero(metrics.get('mate_from_macsa'), 'f1'):.4f}")
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
    if task_enabled(effective_task, "aope"):
        summary.append(f"MAOPE_F1: {metric_or_zero(metrics['aope'], 'f1'):.4f}")
        summary.append(f"MAOPE_PRECISION: {metric_or_zero(metrics['aope'], 'precision'):.4f}")
        summary.append(f"MAOPE_RECALL: {metric_or_zero(metrics['aope'], 'recall'):.4f}")
    log("test_metric - " + " | ".join(summary))

    if task_enabled(effective_task, "mate") and metrics.get("mate_span") is not None:
        log(REPORT_SEPARATOR)
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
        log(REPORT_SEPARATOR)
        log("\n[TEST][MOTE] predicted opinion span vs gold opinion span")
        log(
            f"precision={metrics['mote_span']['precision']:.4f} | "
            f"recall={metrics['mote_span']['recall']:.4f} | "
            f"f1={metrics['mote_span']['f1']:.4f} | "
            f"tp={int(metrics['mote_span']['tp'])} | "
            f"fp={int(metrics['mote_span']['fp'])} | "
            f"fn={int(metrics['mote_span']['fn'])}"
        )
    if task_enabled(effective_task, "macc") and metrics["macc"] is not None:
        log(REPORT_SEPARATOR)
        log("\n[TEST][MACC] classification report")
        log(metrics["macc"]["classification_report"])
    if task_enabled(effective_task, "masc") and metrics["masc"] is not None:
        log(REPORT_SEPARATOR)
        log("\n[TEST][MASC] aspect sentiment classification: sentence + image + aspect -> sentiment")
        log(
            f"accuracy={metrics['masc']['accuracy']:.4f} | "
            f"micro_f1={metrics['masc']['micro_f1']:.4f} | "
            f"correct={int(metrics['masc']['correct'])} | "
            f"total={int(metrics['masc']['total'])}"
        )
        log("\n[TEST][MASC] per-class details")
        log(masc_class_details(metrics["masc"]))
        log(metrics["masc"]["classification_report"])
    if task_enabled(effective_task, "mabsc") and metrics.get("mabsc_span") is not None:
        log(REPORT_SEPARATOR)
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
    if task_enabled(effective_task, "mabsc") and metrics.get("mate_from_mabsc") is not None:
        log(REPORT_SEPARATOR)
        log("\n[TEST][MABSC->MATE] derived aspect span vs gold aspect span")
        log(
            f"precision={metrics['mate_from_mabsc']['precision']:.4f} | "
            f"recall={metrics['mate_from_mabsc']['recall']:.4f} | "
            f"f1={metrics['mate_from_mabsc']['f1']:.4f} | "
            f"tp={int(metrics['mate_from_mabsc']['tp'])} | "
            f"fp={int(metrics['mate_from_mabsc']['fp'])} | "
            f"fn={int(metrics['mate_from_mabsc']['fn'])}"
        )
    if task_enabled(effective_task, "macsa") and metrics.get("macsa_span") is not None:
        log(REPORT_SEPARATOR)
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
    if task_enabled(effective_task, "macsa") and metrics.get("mate_from_macsa") is not None:
        log(REPORT_SEPARATOR)
        log("\n[TEST][MACSA->MATE] derived aspect span vs gold aspect span")
        log(
            f"precision={metrics['mate_from_macsa']['precision']:.4f} | "
            f"recall={metrics['mate_from_macsa']['recall']:.4f} | "
            f"f1={metrics['mate_from_macsa']['f1']:.4f} | "
            f"tp={int(metrics['mate_from_macsa']['tp'])} | "
            f"fp={int(metrics['mate_from_macsa']['fp'])} | "
            f"fn={int(metrics['mate_from_macsa']['fn'])}"
        )
    if task_enabled(effective_task, "quad") and metrics.get("quad") is not None:
        log(REPORT_SEPARATOR)
        log("\n[TEST][QuadPrediction] exact quad set match")
        log(
            f"precision={metrics['quad']['precision']:.4f} | recall={metrics['quad']['recall']:.4f} | "
            f"f1={metrics['quad']['f1']:.4f} | tp={int(metrics['quad']['tp'])} | "
            f"fp={int(metrics['quad']['fp'])} | fn={int(metrics['quad']['fn'])}"
        )
        if metrics["quad"].get("category_sentiment") is not None:
            log(REPORT_SEPARATOR)
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
            log(REPORT_SEPARATOR)
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
        log(REPORT_SEPARATOR)
        log("\n[TEST][QuadPrediction-CLS] exact quad set match from AOPE + MACC + MASC")
        log(
            f"precision={metrics['quad_cls']['precision']:.4f} | recall={metrics['quad_cls']['recall']:.4f} | "
            f"f1={metrics['quad_cls']['f1']:.4f} | tp={int(metrics['quad_cls']['tp'])} | "
            f"fp={int(metrics['quad_cls']['fp'])} | fn={int(metrics['quad_cls']['fn'])}"
        )
        if metrics["quad_cls"].get("category_sentiment") is not None:
            log(REPORT_SEPARATOR)
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
            log(REPORT_SEPARATOR)
            log(f"\n[TEST][QuadPrediction-CLS->{label}] projected from predicted quads")
            log(
                f"precision={projected['precision']:.4f} | "
                f"recall={projected['recall']:.4f} | "
                f"f1={projected['f1']:.4f} | "
                f"tp={int(projected['tp'])} | "
                f"fp={int(projected['fp'])} | "
                f"fn={int(projected['fn'])}"
            )
    if task_enabled(effective_task, "aope") and metrics["aope"] is not None:
        log(REPORT_SEPARATOR)
        log("\n[TEST][MAOPE] relation set match")
        log(
            f"precision={metrics['aope']['precision']:.4f} | recall={metrics['aope']['recall']:.4f} | "
            f"f1={metrics['aope']['f1']:.4f} | tp={int(metrics['aope']['tp'])} | "
            f"fp={int(metrics['aope']['fp'])} | fn={int(metrics['aope']['fn'])}"
        )



