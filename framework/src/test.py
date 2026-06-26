from __future__ import annotations

import os
import argparse
import inspect
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from metric import (
    bio_tags_to_spans,
    evaluate_classification,
    evaluate_quad_sets,
    evaluate_relation_sets,
    evaluate_span_sentiment_sets,
    format_quad_category_sentiment_report,
    format_confusion_matrix,
    mabsc_tags_to_span_sentiments,
    macsa_tags_to_span_categories,
)
from model import MABSABaselineModel
from train import PTDataset, collate_fn, move_to_device


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def safe_evaluate_classification(*args, **kwargs):
    try:
        return evaluate_classification(*args, **kwargs)
    except ValueError as exc:
        if "No valid labels" in str(exc):
            return None
        raise


def model_accepts_arg(model: torch.nn.Module, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _gold_mabsc_sets_from_batch(batch) -> list:
    true_sets = []
    aspect_spans = batch["aspect_spans"]
    sentiments = batch["sentiments"]
    for b_idx, spans in enumerate(aspect_spans):
        items = []
        for s_idx, span in enumerate(spans):
            if b_idx < len(sentiments) and s_idx < len(sentiments[b_idx]):
                items.append((tuple(span), int(sentiments[b_idx][s_idx])))
        true_sets.append(items)
    return true_sets


@torch.no_grad()
def _derive_mabsc_from_mate_masc(model, batch, mate_pred, device: torch.device) -> tuple:
    mask_cpu = batch["attention_mask"].detach().cpu()
    mate_pred_cpu = mate_pred.detach().cpu()
    pred_aspect_spans = [
        bio_tags_to_spans(mate_pred_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist())
        for b_idx in range(mate_pred_cpu.size(0))
    ]

    if not any(pred_aspect_spans):
        return _gold_mabsc_sets_from_batch(batch), [[] for _ in pred_aspect_spans]

    forward_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "image": batch["image"],
        "aspect_spans": pred_aspect_spans,
        "opinion_spans": batch["opinion_spans"],
        "sentiments": batch["sentiments"],
    }
    if model_accepts_arg(model, "mabsc_labels"):
        forward_kwargs["mabsc_labels"] = batch["mabsc_labels"]
    if model_accepts_arg(model, "macsa_labels"):
        forward_kwargs["macsa_labels"] = batch["macsa_labels"]
    outputs = model(**forward_kwargs)
    masc_logits = outputs.get("masc_logits")
    masc_index_map = outputs.get("masc_index_map", [])
    pred_sets = [[] for _ in pred_aspect_spans]
    if masc_logits is not None and len(masc_index_map) > 0:
        masc_pred = masc_logits.argmax(dim=-1).detach().cpu().tolist()
        for i, (b_idx, s_idx) in enumerate(masc_index_map):
            if b_idx < len(pred_sets) and s_idx < len(pred_aspect_spans[b_idx]):
                pred_sets[b_idx].append((tuple(pred_aspect_spans[b_idx][s_idx]), int(masc_pred[i])))

    return _gold_mabsc_sets_from_batch(batch), pred_sets


@torch.no_grad()
def evaluate_test(
    model: MABSABaselineModel,
    dataloader: DataLoader,
    device: torch.device,
    derive_mate_masc_from_mabsc: bool = False,
):
    model.eval()

    mate_true_all, mate_pred_all = [], []
    mate_true_sets, mate_pred_sets = [], []
    mote_true_all, mote_pred_all = [], []
    mote_true_sets, mote_pred_sets = [], []
    macc_true_all, macc_pred_all = [], []
    masc_true_all, masc_pred_all = [], []
    aope_true_sets, aope_pred_sets = [], []
    mabsc_true_all, mabsc_pred_all = [], []
    mabsc_true_sets, mabsc_pred_sets = [], []
    macsa_true_all, macsa_pred_all = [], []
    macsa_true_sets, macsa_pred_sets = [], []
    quad_true_sets, quad_pred_sets = [], []
    quad_cls_true_sets, quad_cls_pred_sets = [], []

    for batch in tqdm(dataloader, desc="Testing", leave=False):
        batch = move_to_device(batch, device)
        forward_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "image": batch["image"],
            "aspect_spans": batch["aspect_spans"],
            "opinion_spans": batch["opinion_spans"],
            "sentiments": batch["sentiments"],
        }
        if model_accepts_arg(model, "mabsc_labels"):
            forward_kwargs["mabsc_labels"] = batch["mabsc_labels"]
        if model_accepts_arg(model, "macsa_labels"):
            forward_kwargs["macsa_labels"] = batch["macsa_labels"]
        outputs = model(**forward_kwargs)

        mate_pred = outputs.get("mate_pred")
        if mate_pred is None:
            mate_pred = outputs["mate_logits"].argmax(dim=-1)
        mote_pred = outputs["mote_logits"].argmax(dim=-1)
        macc_logits = outputs.get("macc_logits")
        macc_index_map = outputs.get("macc_index_map", [])
        batch_macc_pred_lookup = [{} for _ in batch["aspect_spans"]]

        mabsc_logits = outputs.get("mabsc_logits")
        if mabsc_logits is not None:
            mabsc_pred = outputs.get("mabsc_pred")
            if mabsc_pred is None:
                mabsc_pred = mabsc_logits.argmax(dim=-1)
            mabsc_true_all.extend(batch["mabsc_labels"].reshape(-1).detach().cpu().tolist())
            mabsc_pred_all.extend(mabsc_pred.reshape(-1).detach().cpu().tolist())
            mabsc_true_cpu = batch["mabsc_labels"].detach().cpu()
            mabsc_pred_cpu = mabsc_pred.detach().cpu()
            mask_cpu = batch["attention_mask"].detach().cpu()
            for b_idx in range(mabsc_pred_cpu.size(0)):
                mabsc_true_sets.append(
                    mabsc_tags_to_span_sentiments(mabsc_true_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist())
                )
                mabsc_pred_sets.append(
                    mabsc_tags_to_span_sentiments(mabsc_pred_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist())
                )

        macsa_logits = outputs.get("macsa_logits")
        if macsa_logits is not None:
            macsa_pred = macsa_logits.argmax(dim=-1)
            macsa_true_all.extend(batch["macsa_labels"].reshape(-1).detach().cpu().tolist())
            macsa_pred_all.extend(macsa_pred.reshape(-1).detach().cpu().tolist())
            macsa_true_cpu = batch["macsa_labels"].detach().cpu()
            macsa_pred_cpu = macsa_pred.detach().cpu()
            macsa_mask_cpu = batch["attention_mask"].detach().cpu()
            for b_idx in range(macsa_pred_cpu.size(0)):
                macsa_true_sets.append(
                    macsa_tags_to_span_categories(macsa_true_cpu[b_idx].tolist(), macsa_mask_cpu[b_idx].tolist())
                )
                macsa_pred_sets.append(
                    macsa_tags_to_span_categories(macsa_pred_cpu[b_idx].tolist(), macsa_mask_cpu[b_idx].tolist())
                )

        mate_true_all.extend(batch["mate_labels"].reshape(-1).cpu().tolist())
        mate_pred_all.extend(mate_pred.reshape(-1).cpu().tolist())
        mate_true_cpu = batch["mate_labels"].detach().cpu()
        mate_pred_cpu = mate_pred.detach().cpu()
        mask_cpu = batch["attention_mask"].detach().cpu()
        for b_idx in range(mate_pred_cpu.size(0)):
            mate_true_sets.append(bio_tags_to_spans(mate_true_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist()))
            mate_pred_sets.append(bio_tags_to_spans(mate_pred_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist()))
        mote_true_all.extend(batch["mote_labels"].reshape(-1).cpu().tolist())
        mote_pred_all.extend(mote_pred.reshape(-1).cpu().tolist())
        mote_true_cpu = batch["mote_labels"].detach().cpu()
        mote_pred_cpu = mote_pred.detach().cpu()
        for b_idx in range(mote_pred_cpu.size(0)):
            mote_true_sets.append(bio_tags_to_spans(mote_true_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist()))
            mote_pred_sets.append(bio_tags_to_spans(mote_pred_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist()))
        if macc_logits is not None and len(macc_index_map) > 0:
            macc_pred = macc_logits.argmax(dim=-1).cpu().tolist()
            for i, (b_idx, s_idx) in enumerate(macc_index_map):
                if b_idx < len(batch["categories"]) and s_idx < len(batch["categories"][b_idx]):
                    macc_true_all.append(int(batch["categories"][b_idx][s_idx]))
                    macc_pred_all.append(int(macc_pred[i]))
                    if b_idx < len(batch_macc_pred_lookup):
                        batch_macc_pred_lookup[b_idx][int(s_idx)] = int(macc_pred[i])

        masc_logits = outputs.get("masc_logits")
        masc_index_map = outputs.get("masc_index_map", [])
        batch_masc_pred_lookup = [{} for _ in batch["aspect_spans"]]
        if masc_logits is not None and len(masc_index_map) > 0:
            masc_pred = masc_logits.argmax(dim=-1).cpu().tolist()
            for i, (b_idx, s_idx) in enumerate(masc_index_map):
                if b_idx < len(batch["sentiments"]) and s_idx < len(batch["sentiments"][b_idx]):
                    masc_true_all.append(int(batch["sentiments"][b_idx][s_idx]))
                    masc_pred_all.append(int(masc_pred[i]))
                    if b_idx < len(batch_masc_pred_lookup):
                        batch_masc_pred_lookup[b_idx][int(s_idx)] = int(masc_pred[i])

        if mabsc_logits is None and masc_logits is not None:
            derived_true, derived_pred = _derive_mabsc_from_mate_masc(model, batch, mate_pred, device)
            mabsc_true_sets.extend(derived_true)
            mabsc_pred_sets.extend(derived_pred)

        aope_logits = outputs.get("aope_logits")
        aope_index_map = outputs.get("aope_index_map", [])
        batch_aope_pred_sets = [[] for _ in batch["aope_relations"]]
        if aope_logits is not None and len(aope_index_map) > 0:
            aope_pred = aope_logits.argmax(dim=-1).cpu().tolist()
            for i, (b_idx, a_idx, o_idx) in enumerate(aope_index_map):
                if int(aope_pred[i]) == 1 and b_idx < len(batch_aope_pred_sets):
                    batch_aope_pred_sets[b_idx].append((int(a_idx), int(o_idx)))
        aope_true_sets.extend([[tuple(item) for item in items] for items in batch["aope_relations"]])
        aope_pred_sets.extend(batch_aope_pred_sets)

        mabsc_pred_lookup = []
        macsa_pred_lookup = []
        if mabsc_logits is not None and macsa_logits is not None:
            macsa_pred_cpu = macsa_logits.argmax(dim=-1).detach().cpu()
            mask_cpu = batch["attention_mask"].detach().cpu()
            batch_size = macsa_pred_cpu.size(0)
            for b_idx in range(batch_size):
                mabsc_pred_lookup.append(dict(mabsc_pred_sets[-batch_size + b_idx]))
                macsa_pred_lookup.append(dict(macsa_tags_to_span_categories(macsa_pred_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist())))
        for b_idx, true_relations in enumerate(batch["aope_relations"]):
            true_quads = []
            pred_quads = []
            pred_cls_quads = []
            for a_idx, o_idx in true_relations:
                if (
                    a_idx < len(batch["aspect_spans"][b_idx])
                    and o_idx < len(batch["opinion_spans"][b_idx])
                    and a_idx < len(batch["categories"][b_idx])
                    and a_idx < len(batch["sentiments"][b_idx])
                ):
                    true_quads.append(
                        (
                            tuple(batch["aspect_spans"][b_idx][a_idx]),
                            int(batch["categories"][b_idx][a_idx]),
                            tuple(batch["opinion_spans"][b_idx][o_idx]),
                            int(batch["sentiments"][b_idx][a_idx]),
                        )
                    )
            if b_idx < len(batch_aope_pred_sets) and b_idx < len(mabsc_pred_lookup) and b_idx < len(macsa_pred_lookup):
                for a_idx, o_idx in batch_aope_pred_sets[b_idx]:
                    if a_idx < len(batch["aspect_spans"][b_idx]) and o_idx < len(batch["opinion_spans"][b_idx]):
                        aspect_span = tuple(batch["aspect_spans"][b_idx][a_idx])
                        opinion_span = tuple(batch["opinion_spans"][b_idx][o_idx])
                        if aspect_span in mabsc_pred_lookup[b_idx] and aspect_span in macsa_pred_lookup[b_idx]:
                            pred_quads.append(
                                (
                                    aspect_span,
                                    int(macsa_pred_lookup[b_idx][aspect_span]),
                                    opinion_span,
                                    int(mabsc_pred_lookup[b_idx][aspect_span]),
                                )
                            )
            if b_idx < len(batch_aope_pred_sets):
                for a_idx, o_idx in batch_aope_pred_sets[b_idx]:
                    if a_idx < len(batch["aspect_spans"][b_idx]) and o_idx < len(batch["opinion_spans"][b_idx]):
                        aspect_span = tuple(batch["aspect_spans"][b_idx][a_idx])
                        opinion_span = tuple(batch["opinion_spans"][b_idx][o_idx])
                        if a_idx in batch_macc_pred_lookup[b_idx] and a_idx in batch_masc_pred_lookup[b_idx]:
                            pred_cls_quads.append(
                                (
                                    aspect_span,
                                    int(batch_macc_pred_lookup[b_idx][a_idx]),
                                    opinion_span,
                                    int(batch_masc_pred_lookup[b_idx][a_idx]),
                                )
                            )
            quad_true_sets.append(true_quads)
            quad_pred_sets.append(pred_quads)
            quad_cls_true_sets.append(true_quads)
            quad_cls_pred_sets.append(pred_cls_quads)

    mate = safe_evaluate_classification(
        mate_true_all, mate_pred_all, label_names=["O", "B-ASP", "I-ASP"], ignore_label=-100
    )
    mate_span = evaluate_relation_sets(mate_true_sets, mate_pred_sets) if len(mate_true_sets) > 0 else None
    mote = safe_evaluate_classification(
        mote_true_all, mote_pred_all, label_names=["O", "B-OPN", "I-OPN"], ignore_label=-100
    )
    mote_span = evaluate_relation_sets(mote_true_sets, mote_pred_sets) if len(mote_true_sets) > 0 else None
    macc = safe_evaluate_classification(
        macc_true_all,
        macc_pred_all,
        label_names=["Facility", "Service", "Amenity", "Experience", "Branding", "Loyalty"],
        ignore_label=-100,
    )
    masc = (
        safe_evaluate_classification(
            masc_true_all, masc_pred_all, label_names=["Negative", "Neutral", "Positive"]
        )
        if len(masc_true_all) > 0
        else None
    )
    aope = evaluate_relation_sets(aope_true_sets, aope_pred_sets) if len(aope_true_sets) > 0 else None
    mabsc = (
        safe_evaluate_classification(
            mabsc_true_all,
            mabsc_pred_all,
            label_names=["O", "B-NEG", "B-NEU", "B-POS", "I"],
            ignore_label=-100,
        )
        if len(mabsc_true_all) > 0
        else None
    )
    mabsc_span = (
        evaluate_span_sentiment_sets(mabsc_true_sets, mabsc_pred_sets)
        if len(mabsc_true_sets) > 0
        else None
    )
    mate_from_mabsc = None
    masc_from_mabsc = None
    if len(mabsc_true_sets) > 0:
        derived_mate_true = [[tuple(span) for span, _ in items] for items in mabsc_true_sets]
        derived_mate_pred = [[tuple(span) for span, _ in items] for items in mabsc_pred_sets]
        mate_from_mabsc = evaluate_relation_sets(derived_mate_true, derived_mate_pred)

        derived_masc_true_all = []
        derived_masc_pred_all = []
        missing_sentiment_label = 3
        for true_items, pred_items in zip(mabsc_true_sets, mabsc_pred_sets):
            pred_by_span = {tuple(span): int(sentiment) for span, sentiment in pred_items}
            for span, sentiment in true_items:
                span = tuple(span)
                derived_masc_true_all.append(int(sentiment))
                derived_masc_pred_all.append(pred_by_span.get(span, missing_sentiment_label))
        masc_from_mabsc = (
            safe_evaluate_classification(
                derived_masc_true_all,
                derived_masc_pred_all,
                label_names=["Negative", "Neutral", "Positive", "Missing"],
            )
            if len(derived_masc_true_all) > 0
            else None
        )
    if derive_mate_masc_from_mabsc and len(mabsc_true_sets) > 0:
        mate_span = mate_from_mabsc
        masc = masc_from_mabsc
    macsa = (
        safe_evaluate_classification(
            macsa_true_all,
            macsa_pred_all,
            label_names=["O", "B-FAC", "B-SER", "B-AME", "B-EXP", "B-BRA", "B-LOY", "I"],
            ignore_label=-100,
        )
        if len(macsa_true_all) > 0
        else None
    )
    macsa_span = (
        evaluate_span_sentiment_sets(macsa_true_sets, macsa_pred_sets)
        if len(macsa_true_sets) > 0
        else None
    )
    mate_from_macsa = None
    macc_from_macsa = None
    if len(macsa_true_sets) > 0:
        derived_mate_true = [[tuple(span) for span, _ in items] for items in macsa_true_sets]
        derived_mate_pred = [[tuple(span) for span, _ in items] for items in macsa_pred_sets]
        mate_from_macsa = evaluate_relation_sets(derived_mate_true, derived_mate_pred)

        derived_macc_true_all = []
        derived_macc_pred_all = []
        missing_category_label = 6
        for true_items, pred_items in zip(macsa_true_sets, macsa_pred_sets):
            pred_by_span = {tuple(span): int(category) for span, category in pred_items}
            for span, category in true_items:
                span = tuple(span)
                derived_macc_true_all.append(int(category))
                derived_macc_pred_all.append(pred_by_span.get(span, missing_category_label))
        macc_from_macsa = (
            safe_evaluate_classification(
                derived_macc_true_all,
                derived_macc_pred_all,
                label_names=["Facility", "Service", "Amenity", "Experience", "Branding", "Loyalty", "Missing"],
            )
            if len(derived_macc_true_all) > 0
            else None
        )
    quad = evaluate_quad_sets(quad_true_sets, quad_pred_sets) if len(quad_true_sets) > 0 else None
    quad_cls = evaluate_quad_sets(quad_cls_true_sets, quad_cls_pred_sets) if len(quad_cls_true_sets) > 0 else None
    quad_projected = quad_projection_metrics(quad_true_sets, quad_pred_sets) if len(quad_true_sets) > 0 else {}
    quad_cls_projected = (
        quad_projection_metrics(quad_cls_true_sets, quad_cls_pred_sets)
        if len(quad_cls_true_sets) > 0
        else {}
    )
    return {
        "mate": mate,
        "mate_span": mate_span,
        "mote": mote,
        "mote_span": mote_span,
        "macc": macc,
        "masc": masc,
        "aope": aope,
        "mabsc": mabsc,
        "mabsc_span": mabsc_span,
        "mate_from_mabsc": mate_from_mabsc,
        "masc_from_mabsc": masc_from_mabsc,
        "macsa": macsa,
        "macsa_span": macsa_span,
        "mate_from_macsa": mate_from_macsa,
        "macc_from_macsa": macc_from_macsa,
        "quad": quad,
        "quad_cls": quad_cls,
        "quad_projected": quad_projected,
        "quad_cls_projected": quad_cls_projected,
    }


def iou_details_from_confusion(metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    labels = metrics["label_names"]
    lines = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        inter = tp
        union = tp + fp + fn
        iou = (inter / union) if union > 0 else 0.0
        lines.append(f"{label}: intersect={inter}, union={union}, iou={iou:.4f}")

    micro_inter = int(np.trace(cm))
    micro_union = int(cm.sum())
    micro_iou = (micro_inter / micro_union) if micro_union > 0 else 0.0
    lines.append(f"micro: intersect={micro_inter}, union={micro_union}, iou={micro_iou:.4f}")
    return "\n".join(lines)


def metric_or_zero(metrics, key: str) -> float:
    return float(metrics[key]) if metrics is not None else 0.0


def quad_component_details(metrics: dict) -> str:
    components = metrics.get("components", {}) if metrics is not None else {}
    display_names = [
        ("aspect", "Aspect span"),
        ("opinion", "Opinion span"),
        ("category", "Category"),
        ("sentiment", "Sentiment"),
        ("aspect_opinion", "Aspect+Opinion"),
        ("aspect_category", "Aspect+Category"),
        ("aspect_sentiment", "Aspect+Sentiment"),
        ("opinion_sentiment", "Opinion+Sentiment"),
        ("aspect_opinion_category", "Aspect+Opinion+Category"),
        ("aspect_opinion_sentiment", "Aspect+Opinion+Sentiment"),
        ("aspect_category_sentiment", "Aspect+Category+Sentiment"),
    ]
    lines = ["component".ljust(28) + "precision  recall     f1        tp     fp     fn"]
    for key, label in display_names:
        item = components.get(key)
        if item is None:
            continue
        lines.append(
            f"{label.ljust(28)}"
            f"{item['precision']:<10.4f}"
            f"{item['recall']:<10.4f}"
            f"{item['f1']:<10.4f}"
            f"{int(item['tp']):<7}"
            f"{int(item['fp']):<7}"
            f"{int(item['fn']):<7}"
        )
    return "\n".join(lines)


def masc_class_details(metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    names = metrics["label_names"]
    lines = ["class                      precision  recall     f1        correct  total"]
    per_class = {item["label_name"]: item for item in metrics.get("per_class", [])}
    for idx, name in enumerate(names):
        item = per_class.get(name, {})
        correct = int(cm[idx, idx])
        total = int(cm[idx, :].sum())
        lines.append(
            f"{name:<26} "
            f"{float(item.get('precision', 0.0)):<10.4f} "
            f"{float(item.get('recall', 0.0)):<10.4f} "
            f"{float(item.get('f1', 0.0)):<9.4f} "
            f"{correct:<8} {total:<8}"
        )
    return "\n".join(lines)


def quad_projection_metrics(quad_true_sets, quad_pred_sets) -> dict:
    true_mate = [[item[0] for item in items] for items in quad_true_sets]
    pred_mate = [[item[0] for item in items] for items in quad_pred_sets]
    true_mote = [[item[2] for item in items] for items in quad_true_sets]
    pred_mote = [[item[2] for item in items] for items in quad_pred_sets]
    true_maope = [[(item[0], item[2]) for item in items] for items in quad_true_sets]
    pred_maope = [[(item[0], item[2]) for item in items] for items in quad_pred_sets]
    true_mabsc = [[(item[0], item[3]) for item in items] for items in quad_true_sets]
    pred_mabsc = [[(item[0], item[3]) for item in items] for items in quad_pred_sets]
    true_macsa = [[(item[0], item[1]) for item in items] for items in quad_true_sets]
    pred_macsa = [[(item[0], item[1]) for item in items] for items in quad_pred_sets]
    return {
        "mate": evaluate_relation_sets(true_mate, pred_mate),
        "mote": evaluate_relation_sets(true_mote, pred_mote),
        "maope": evaluate_relation_sets(true_maope, pred_maope),
        "mabsc": evaluate_span_sentiment_sets(true_mabsc, pred_mabsc),
        "macsa": evaluate_span_sentiment_sets(true_macsa, pred_macsa),
    }


def task_enabled(task: str, name: str) -> bool:
    names = {x.strip().lower() for x in task.split(",") if x.strip()}
    if "asqp" in names or "quadra" in names:
        names.update({"mate", "mote", "maope", "aope", "mabsc", "macsa", "quad"})
    if "quadprediction" in names or "quad" in names:
        names.update({"mate", "mote", "maope", "aope", "mabsc", "macsa", "quad"})
    if "quad_cls" in names or "quadclass" in names or "quad_classification" in names:
        names.update({"maope", "aope", "macc", "masc", "quad_cls"})
    if "maope" in names:
        names.add("aope")
    return name in names


REPORT_SEPARATOR = "=" * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MABSA baseline model on test set")
    parser.add_argument("--test_dir", type=str, default="formatted_data/test")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--vision_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--num_categories", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--task", type=str, default="mate,mote,macc,masc")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    TEST_DIR = args.test_dir
    CKPT_PATH = args.ckpt_path
    LOG_DIR = args.log_dir

    TEXT_MODEL_NAME = args.text_model_name
    VISION_MODEL_NAME = args.vision_model_name
    NUM_CATEGORIES = args.num_categories
    BATCH_SIZE = args.batch_size
    DEVICE = args.device

    os.makedirs(LOG_DIR, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{run_time}_test.log")

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = torch.device(DEVICE)
    test_ds = PTDataset(TEST_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = MABSABaselineModel(
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        num_categories=NUM_CATEGORIES,
    ).to(device)
    state = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)

    log(f"Log file: {log_path}")
    log(f"Device: {DEVICE}")
    log("Config:")
    log(f"  TEST_DIR={TEST_DIR}")
    log(f"  CKPT_PATH={CKPT_PATH}")
    log(f"  BATCH_SIZE={BATCH_SIZE}")
    log(f"  TEXT_MODEL_NAME={TEXT_MODEL_NAME}")
    log(f"  VISION_MODEL_NAME={VISION_MODEL_NAME}")
    log(f"  NUM_CATEGORIES={NUM_CATEGORIES}")
    log(f"  TASK={args.task}")
    log(f"Test samples: {len(test_ds)}")

    metrics = evaluate_test(model, test_loader, device)
    summary = []
    if task_enabled(args.task, "mate"):
        summary.append(f"MATE_SPAN_F1: {metric_or_zero(metrics.get('mate_span'), 'f1'):.4f}")
        summary.append(f"MATE_SPAN_PRECISION: {metric_or_zero(metrics.get('mate_span'), 'precision'):.4f}")
        summary.append(f"MATE_SPAN_RECALL: {metric_or_zero(metrics.get('mate_span'), 'recall'):.4f}")
    if task_enabled(args.task, "mote"):
        summary.append(f"MOTE_SPAN_F1: {metric_or_zero(metrics.get('mote_span'), 'f1'):.4f}")
        summary.append(f"MOTE_SPAN_PRECISION: {metric_or_zero(metrics.get('mote_span'), 'precision'):.4f}")
        summary.append(f"MOTE_SPAN_RECALL: {metric_or_zero(metrics.get('mote_span'), 'recall'):.4f}")
    if task_enabled(args.task, "macc"):
        summary.append(f"MACC_MICRO_F1: {metric_or_zero(metrics['macc'], 'micro_f1'):.4f}")
        summary.append(f"MACC_MACRO_F1: {metric_or_zero(metrics['macc'], 'macro_f1'):.4f}")
    if task_enabled(args.task, "masc"):
        summary.append(f"MASC_ACCURACY: {metric_or_zero(metrics['masc'], 'accuracy'):.4f}")
        summary.append(f"MASC_MICRO_F1: {metric_or_zero(metrics['masc'], 'micro_f1'):.4f}")
        summary.append(f"MASC_CORRECT: {int(metric_or_zero(metrics['masc'], 'correct'))}")
        summary.append(f"MASC_TOTAL: {int(metric_or_zero(metrics['masc'], 'total'))}")
        summary.append(f"MASC_MACRO_F1: {metric_or_zero(metrics['masc'], 'macro_f1'):.4f}")
    if task_enabled(args.task, "aope"):
        summary.append(f"MAOPE_F1: {metric_or_zero(metrics['aope'], 'f1'):.4f}")
        summary.append(f"MAOPE_PRECISION: {metric_or_zero(metrics['aope'], 'precision'):.4f}")
        summary.append(f"MAOPE_RECALL: {metric_or_zero(metrics['aope'], 'recall'):.4f}")
    if task_enabled(args.task, "mabsc"):
        summary.append(f"MABSC_SPAN_F1: {metric_or_zero(metrics.get('mabsc_span'), 'f1'):.4f}")
        summary.append(f"MABSC_SPAN_PRECISION: {metric_or_zero(metrics.get('mabsc_span'), 'precision'):.4f}")
        summary.append(f"MABSC_SPAN_RECALL: {metric_or_zero(metrics.get('mabsc_span'), 'recall'):.4f}")
        summary.append(f"MABSC_AS_MATE_F1: {metric_or_zero(metrics.get('mate_from_mabsc'), 'f1'):.4f}")
    if task_enabled(args.task, "macsa"):
        summary.append(f"MACSA_SPAN_F1: {metric_or_zero(metrics.get('macsa_span'), 'f1'):.4f}")
        summary.append(f"MACSA_SPAN_PRECISION: {metric_or_zero(metrics.get('macsa_span'), 'precision'):.4f}")
        summary.append(f"MACSA_SPAN_RECALL: {metric_or_zero(metrics.get('macsa_span'), 'recall'):.4f}")
        summary.append(f"MACSA_AS_MATE_F1: {metric_or_zero(metrics.get('mate_from_macsa'), 'f1'):.4f}")
    if task_enabled(args.task, "quad"):
        summary.append(f"QUAD_F1: {metric_or_zero(metrics.get('quad'), 'f1'):.4f}")
        summary.append(f"QUAD_PRECISION: {metric_or_zero(metrics.get('quad'), 'precision'):.4f}")
        summary.append(f"QUAD_RECALL: {metric_or_zero(metrics.get('quad'), 'recall'):.4f}")
        summary.append(f"QUAD_AS_MABSC_F1: {metric_or_zero(metrics.get('quad_projected', {}).get('mabsc'), 'f1'):.4f}")
        summary.append(f"QUAD_AS_MACSA_F1: {metric_or_zero(metrics.get('quad_projected', {}).get('macsa'), 'f1'):.4f}")
    if task_enabled(args.task, "quad_cls"):
        summary.append(f"QUAD_CLS_F1: {metric_or_zero(metrics.get('quad_cls'), 'f1'):.4f}")
        summary.append(f"QUAD_CLS_PRECISION: {metric_or_zero(metrics.get('quad_cls'), 'precision'):.4f}")
        summary.append(f"QUAD_CLS_RECALL: {metric_or_zero(metrics.get('quad_cls'), 'recall'):.4f}")
        summary.append(f"QUAD_CLS_AS_MABSC_F1: {metric_or_zero(metrics.get('quad_cls_projected', {}).get('mabsc'), 'f1'):.4f}")
        summary.append(f"QUAD_CLS_AS_MACSA_F1: {metric_or_zero(metrics.get('quad_cls_projected', {}).get('macsa'), 'f1'):.4f}")
    log("test_metric - " + " | ".join(summary))

    if task_enabled(args.task, "mate") and metrics.get("mate_span") is not None:
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

    if task_enabled(args.task, "mote") and metrics.get("mote_span") is not None:
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

    if task_enabled(args.task, "macc") and metrics["macc"] is not None:
        log(REPORT_SEPARATOR)
        log("\n[TEST][MACC] classification report")
        log(metrics["macc"]["classification_report"])

    if task_enabled(args.task, "masc") and metrics["masc"] is not None:
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
    if task_enabled(args.task, "aope") and metrics["aope"] is not None:
        log(REPORT_SEPARATOR)
        log("\n[TEST][MAOPE] relation set match")
        log(
            f"precision={metrics['aope']['precision']:.4f} | recall={metrics['aope']['recall']:.4f} | "
            f"f1={metrics['aope']['f1']:.4f} | tp={int(metrics['aope']['tp'])} | "
            f"fp={int(metrics['aope']['fp'])} | fn={int(metrics['aope']['fn'])}"
        )
    if task_enabled(args.task, "mabsc") and metrics.get("mabsc_span") is not None:
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
    if task_enabled(args.task, "mabsc") and metrics.get("mate_from_mabsc") is not None:
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
    if task_enabled(args.task, "macsa") and metrics.get("macsa_span") is not None:
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
    if task_enabled(args.task, "macsa") and metrics.get("mate_from_macsa") is not None:
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
    if task_enabled(args.task, "quad") and metrics.get("quad") is not None:
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
    if task_enabled(args.task, "quad_cls") and metrics.get("quad_cls") is not None:
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



