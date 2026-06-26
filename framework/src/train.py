from __future__ import annotations

import os
import argparse
import inspect
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metric import (
    bio_tags_to_spans,
    evaluate_classification,
    evaluate_quad_sets,
    evaluate_relation_sets,
    evaluate_span_sentiment_sets,
    format_confusion_matrix,
    mabsc_tags_to_span_sentiments,
    macsa_tags_to_span_categories,
)
from model import MABSABaselineModel


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


class PTDataset(Dataset):
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.files = sorted([p for p in self.data_dir.glob("*.pt") if p.is_file()])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return sample


def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    has_semantic_text = all(("input_ids_semantic" in x and "attention_mask_semantic" in x) for x in batch)
    input_ids_semantic = (
        torch.stack([x["input_ids_semantic"] for x in batch], dim=0) if has_semantic_text else None
    )
    attention_mask_semantic = (
        torch.stack([x["attention_mask_semantic"] for x in batch], dim=0) if has_semantic_text else None
    )
    mate_labels = torch.stack([x["mate_labels"] for x in batch], dim=0)
    mote_labels = torch.stack([x["mote_labels"] for x in batch], dim=0)
    categories = [x.get("categories", []) for x in batch]

    images: List[torch.Tensor] = []
    has_all_images = True
    for x in batch:
        if x.get("image") is None or not isinstance(x.get("image"), torch.Tensor):
            has_all_images = False
            break
        images.append(x["image"])
    image = torch.stack(images, dim=0) if has_all_images else None

    semantic_images: List[torch.Tensor] = []
    has_all_semantic_images = True
    for x in batch:
        if x.get("image_semantic") is None or not isinstance(x.get("image_semantic"), torch.Tensor):
            has_all_semantic_images = False
            break
        semantic_images.append(x["image_semantic"])
    image_semantic = torch.stack(semantic_images, dim=0) if has_all_semantic_images else None

    aspect_spans = [x.get("aspect_spans", []) for x in batch]
    opinion_spans = [x.get("opinion_spans", []) for x in batch]
    legacy_ao_key = "ao_" + "".join(["p", "a", "i", "r", "s"])
    aope_relations = [x.get("aope_relations", x.get(legacy_ao_key, [])) for x in batch]
    sentiments = [x.get("sentiments", []) for x in batch]
    mabsc_labels = torch.zeros_like(mate_labels)
    mabsc_labels = mabsc_labels.masked_fill(mate_labels == -100, -100)
    macsa_labels = torch.zeros_like(mate_labels)
    macsa_labels = macsa_labels.masked_fill(mate_labels == -100, -100)
    for b_idx, spans in enumerate(aspect_spans):
        for s_idx, span in enumerate(spans):
            start, end = int(span[0]), int(span[1])
            start = max(0, min(start, mabsc_labels.size(1)))
            end = max(start, min(end, mabsc_labels.size(1)))
            if end <= start:
                continue
            if b_idx < len(sentiments) and s_idx < len(sentiments[b_idx]):
                sentiment = int(sentiments[b_idx][s_idx])
                if 0 <= sentiment <= 2:
                    mabsc_labels[b_idx, start] = 1 + sentiment
                    if end > start + 1:
                        mabsc_labels[b_idx, start + 1 : end] = 4
            if b_idx < len(categories) and s_idx < len(categories[b_idx]):
                category = int(categories[b_idx][s_idx])
                if 0 <= category <= 5:
                    macsa_labels[b_idx, start] = 1 + category
                    if end > start + 1:
                        macsa_labels[b_idx, start + 1 : end] = 7

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_ids_semantic": input_ids_semantic,
        "attention_mask_semantic": attention_mask_semantic,
        "mate_labels": mate_labels,
        "mote_labels": mote_labels,
        "mabsc_labels": mabsc_labels,
        "macsa_labels": macsa_labels,
        "categories": categories,
        "image": image,
        "image_semantic": image_semantic,
        "aspect_spans": aspect_spans,
        "opinion_spans": opinion_spans,
        "aope_relations": aope_relations,
        "sentiments": sentiments,
    }


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    for key in ["input_ids", "attention_mask", "mate_labels", "mote_labels", "mabsc_labels", "macsa_labels"]:
        out[key] = out[key].to(device)
    if out.get("input_ids_semantic") is not None:
        out["input_ids_semantic"] = out["input_ids_semantic"].to(device)
    if out.get("attention_mask_semantic") is not None:
        out["attention_mask_semantic"] = out["attention_mask_semantic"].to(device)
    if out["image"] is not None:
        out["image"] = out["image"].to(device)
    if out.get("image_semantic") is not None:
        out["image_semantic"] = out["image_semantic"].to(device)
    return out


@torch.no_grad()
def evaluate(
    model: MABSABaselineModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    mate_true_all: List[int] = []
    mate_pred_all: List[int] = []
    mate_true_sets: List[List[Tuple[int, int]]] = []
    mate_pred_sets: List[List[Tuple[int, int]]] = []
    mote_true_all: List[int] = []
    mote_pred_all: List[int] = []
    mote_true_sets: List[List[Tuple[int, int]]] = []
    mote_pred_sets: List[List[Tuple[int, int]]] = []
    macc_true_all: List[int] = []
    macc_pred_all: List[int] = []
    masc_true_all: List[int] = []
    masc_pred_all: List[int] = []
    aope_true_sets: List[List[Tuple[int, int]]] = []
    aope_pred_sets: List[List[Tuple[int, int]]] = []
    mabsc_true_all: List[int] = []
    mabsc_pred_all: List[int] = []
    mabsc_true_sets: List[List[Tuple[Tuple[int, int], int]]] = []
    mabsc_pred_sets: List[List[Tuple[Tuple[int, int], int]]] = []
    macsa_true_all: List[int] = []
    macsa_pred_all: List[int] = []
    macsa_true_sets: List[List[Tuple[Tuple[int, int], int]]] = []
    macsa_pred_sets: List[List[Tuple[Tuple[int, int], int]]] = []
    quad_true_sets: List[List[Tuple[Tuple[int, int], int, Tuple[int, int], int]]] = []
    quad_pred_sets: List[List[Tuple[Tuple[int, int], int, Tuple[int, int], int]]] = []
    quad_cls_true_sets: List[List[Tuple[Tuple[int, int], int, Tuple[int, int], int]]] = []
    quad_cls_pred_sets: List[List[Tuple[Tuple[int, int], int, Tuple[int, int], int]]] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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
        batch_macc_pred_lookup: List[Dict[int, int]] = [{} for _ in batch["aspect_spans"]]

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

        mate_true_all.extend(batch["mate_labels"].reshape(-1).detach().cpu().tolist())
        mate_pred_all.extend(mate_pred.reshape(-1).detach().cpu().tolist())
        mate_true_cpu = batch["mate_labels"].detach().cpu()
        mate_pred_cpu = mate_pred.detach().cpu()
        mask_cpu = batch["attention_mask"].detach().cpu()
        for b_idx in range(mate_pred_cpu.size(0)):
            mate_true_sets.append(bio_tags_to_spans(mate_true_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist()))
            mate_pred_sets.append(bio_tags_to_spans(mate_pred_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist()))
        mote_true_all.extend(batch["mote_labels"].reshape(-1).detach().cpu().tolist())
        mote_pred_all.extend(mote_pred.reshape(-1).detach().cpu().tolist())
        mote_true_cpu = batch["mote_labels"].detach().cpu()
        mote_pred_cpu = mote_pred.detach().cpu()
        for b_idx in range(mote_pred_cpu.size(0)):
            mote_true_sets.append(bio_tags_to_spans(mote_true_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist()))
            mote_pred_sets.append(bio_tags_to_spans(mote_pred_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist()))
        if macc_logits is not None and len(macc_index_map) > 0:
            macc_pred = macc_logits.argmax(dim=-1).detach().cpu().tolist()
            for i, (b_idx, s_idx) in enumerate(macc_index_map):
                if b_idx < len(batch["categories"]) and s_idx < len(batch["categories"][b_idx]):
                    macc_true_all.append(int(batch["categories"][b_idx][s_idx]))
                    macc_pred_all.append(int(macc_pred[i]))
                    if b_idx < len(batch_macc_pred_lookup):
                        batch_macc_pred_lookup[b_idx][int(s_idx)] = int(macc_pred[i])

        masc_logits = outputs.get("masc_logits")
        masc_index_map = outputs.get("masc_index_map", [])
        batch_masc_pred_lookup: List[Dict[int, int]] = [{} for _ in batch["aspect_spans"]]
        if masc_logits is not None and len(masc_index_map) > 0:
            masc_pred = masc_logits.argmax(dim=-1).detach().cpu().tolist()
            for i, (b_idx, s_idx) in enumerate(masc_index_map):
                if b_idx < len(batch["sentiments"]) and s_idx < len(batch["sentiments"][b_idx]):
                    masc_true_all.append(int(batch["sentiments"][b_idx][s_idx]))
                    masc_pred_all.append(int(masc_pred[i]))
                    if b_idx < len(batch_masc_pred_lookup):
                        batch_masc_pred_lookup[b_idx][int(s_idx)] = int(masc_pred[i])

        aope_logits = outputs.get("aope_logits")
        aope_index_map = outputs.get("aope_index_map", [])
        batch_aope_pred_sets: List[List[Tuple[int, int]]] = [[] for _ in batch["aope_relations"]]
        if aope_logits is not None and len(aope_index_map) > 0:
            aope_pred = aope_logits.argmax(dim=-1).detach().cpu().tolist()
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
            for b_idx in range(macsa_pred_cpu.size(0)):
                mabsc_pred_lookup.append(dict(mabsc_pred_sets[-macsa_pred_cpu.size(0) + b_idx]))
                macsa_pred_lookup.append(dict(macsa_tags_to_span_categories(macsa_pred_cpu[b_idx].tolist(), mask_cpu[b_idx].tolist())))
        for b_idx, true_relations in enumerate(batch["aope_relations"]):
            true_quads: List[Tuple[Tuple[int, int], int, Tuple[int, int], int]] = []
            pred_quads: List[Tuple[Tuple[int, int], int, Tuple[int, int], int]] = []
            pred_cls_quads: List[Tuple[Tuple[int, int], int, Tuple[int, int], int]] = []
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

    mate_metrics = safe_evaluate_classification(
        mate_true_all, mate_pred_all, label_names=["O", "B-ASP", "I-ASP"], ignore_label=-100
    )
    mate_span_metrics = evaluate_relation_sets(mate_true_sets, mate_pred_sets) if len(mate_true_sets) > 0 else None
    mote_metrics = safe_evaluate_classification(
        mote_true_all, mote_pred_all, label_names=["O", "B-OPN", "I-OPN"], ignore_label=-100
    )
    mote_span_metrics = evaluate_relation_sets(mote_true_sets, mote_pred_sets) if len(mote_true_sets) > 0 else None
    macc_metrics = safe_evaluate_classification(
        macc_true_all,
        macc_pred_all,
        label_names=["Facility", "Service", "Amenity", "Experience", "Branding", "Loyalty"],
        ignore_label=-100,
    )
    masc_metrics = (
        safe_evaluate_classification(
            masc_true_all, masc_pred_all, label_names=["Negative", "Neutral", "Positive"]
        )
        if len(masc_true_all) > 0
        else None
    )
    aope_metrics = evaluate_relation_sets(aope_true_sets, aope_pred_sets) if len(aope_true_sets) > 0 else None
    mabsc_metrics = (
        safe_evaluate_classification(
            mabsc_true_all,
            mabsc_pred_all,
            label_names=["O", "B-NEG", "B-NEU", "B-POS", "I"],
            ignore_label=-100,
        )
        if len(mabsc_true_all) > 0
        else None
    )
    mabsc_span_metrics = (
        evaluate_span_sentiment_sets(mabsc_true_sets, mabsc_pred_sets)
        if len(mabsc_true_sets) > 0
        else None
    )
    macsa_metrics = (
        safe_evaluate_classification(
            macsa_true_all,
            macsa_pred_all,
            label_names=["O", "B-FAC", "B-SER", "B-AME", "B-EXP", "B-BRA", "B-LOY", "I"],
            ignore_label=-100,
        )
        if len(macsa_true_all) > 0
        else None
    )
    macsa_span_metrics = (
        evaluate_span_sentiment_sets(macsa_true_sets, macsa_pred_sets)
        if len(macsa_true_sets) > 0
        else None
    )
    quad_metrics = evaluate_quad_sets(quad_true_sets, quad_pred_sets) if len(quad_true_sets) > 0 else None
    quad_cls_metrics = evaluate_quad_sets(quad_cls_true_sets, quad_cls_pred_sets) if len(quad_cls_true_sets) > 0 else None

    return {
        "mate": mate_metrics,
        "mate_span": mate_span_metrics,
        "mote": mote_metrics,
        "mote_span": mote_span_metrics,
        "macc": macc_metrics,
        "masc": masc_metrics,
        "aope": aope_metrics,
        "mabsc": mabsc_metrics,
        "mabsc_span": mabsc_span_metrics,
        "macsa": macsa_metrics,
        "macsa_span": macsa_span_metrics,
        "quad": quad_metrics,
        "quad_cls": quad_cls_metrics,
    }


def get_score_summary(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "mate_iou": metrics["mate"]["macro_iou"] if metrics["mate"] is not None else 0.0,
        "mate_micro_f1": metrics["mate"]["micro_f1"] if metrics["mate"] is not None else 0.0,
        "mate_macro_f1": metrics["mate"]["macro_f1"] if metrics["mate"] is not None else 0.0,
        "mate_span_f1": metrics["mate_span"]["f1"] if metrics.get("mate_span") is not None else 0.0,
        "mate_span_precision": metrics["mate_span"]["precision"] if metrics.get("mate_span") is not None else 0.0,
        "mate_span_recall": metrics["mate_span"]["recall"] if metrics.get("mate_span") is not None else 0.0,
        "mote_iou": metrics["mote"]["macro_iou"] if metrics["mote"] is not None else 0.0,
        "mote_micro_f1": metrics["mote"]["micro_f1"] if metrics["mote"] is not None else 0.0,
        "mote_macro_f1": metrics["mote"]["macro_f1"] if metrics["mote"] is not None else 0.0,
        "mote_span_f1": metrics["mote_span"]["f1"] if metrics.get("mote_span") is not None else 0.0,
        "mote_span_precision": metrics["mote_span"]["precision"] if metrics.get("mote_span") is not None else 0.0,
        "mote_span_recall": metrics["mote_span"]["recall"] if metrics.get("mote_span") is not None else 0.0,
        "macc_micro_f1": metrics["macc"]["micro_f1"] if metrics["macc"] is not None else 0.0,
        "macc_macro_f1": metrics["macc"]["macro_f1"] if metrics["macc"] is not None else 0.0,
        "masc_accuracy": metrics["masc"]["accuracy"] if metrics["masc"] is not None else 0.0,
        "masc_correct": metrics["masc"]["correct"] if metrics["masc"] is not None else 0.0,
        "masc_total": metrics["masc"]["total"] if metrics["masc"] is not None else 0.0,
        "masc_micro_f1": metrics["masc"]["micro_f1"] if metrics["masc"] is not None else 0.0,
        "masc_macro_f1": metrics["masc"]["macro_f1"] if metrics["masc"] is not None else 0.0,
        "aope_micro_f1": metrics["aope"]["f1"] if metrics["aope"] is not None else 0.0,
        "aope_macro_f1": metrics["aope"]["f1"] if metrics["aope"] is not None else 0.0,
        "aope_precision": metrics["aope"]["precision"] if metrics["aope"] is not None else 0.0,
        "aope_recall": metrics["aope"]["recall"] if metrics["aope"] is not None else 0.0,
        "mabsc_micro_f1": metrics["mabsc"]["micro_f1"] if metrics.get("mabsc") is not None else 0.0,
        "mabsc_macro_f1": metrics["mabsc"]["macro_f1"] if metrics.get("mabsc") is not None else 0.0,
        "mabsc_span_f1": metrics["mabsc_span"]["f1"] if metrics.get("mabsc_span") is not None else 0.0,
        "mabsc_span_precision": metrics["mabsc_span"]["precision"] if metrics.get("mabsc_span") is not None else 0.0,
        "mabsc_span_recall": metrics["mabsc_span"]["recall"] if metrics.get("mabsc_span") is not None else 0.0,
        "mabsc_span_only_fp": metrics["mabsc_span"]["only_fp"] if metrics.get("mabsc_span") is not None else 0.0,
        "mabsc_span_only_fn": metrics["mabsc_span"]["only_fn"] if metrics.get("mabsc_span") is not None else 0.0,
        "mabsc_span_fp_fn": metrics["mabsc_span"]["fp_fn"] if metrics.get("mabsc_span") is not None else 0.0,
        "mabsc_span_exact_match": metrics["mabsc_span"]["exact_match"] if metrics.get("mabsc_span") is not None else 0.0,
        "macsa_micro_f1": metrics["macsa"]["micro_f1"] if metrics.get("macsa") is not None else 0.0,
        "macsa_macro_f1": metrics["macsa"]["macro_f1"] if metrics.get("macsa") is not None else 0.0,
        "macsa_span_f1": metrics["macsa_span"]["f1"] if metrics.get("macsa_span") is not None else 0.0,
        "macsa_span_precision": metrics["macsa_span"]["precision"] if metrics.get("macsa_span") is not None else 0.0,
        "macsa_span_recall": metrics["macsa_span"]["recall"] if metrics.get("macsa_span") is not None else 0.0,
        "quad_f1": metrics["quad"]["f1"] if metrics.get("quad") is not None else 0.0,
        "quad_precision": metrics["quad"]["precision"] if metrics.get("quad") is not None else 0.0,
        "quad_recall": metrics["quad"]["recall"] if metrics.get("quad") is not None else 0.0,
        "quad_cls_f1": metrics["quad_cls"]["f1"] if metrics.get("quad_cls") is not None else 0.0,
        "quad_cls_precision": metrics["quad_cls"]["precision"] if metrics.get("quad_cls") is not None else 0.0,
        "quad_cls_recall": metrics["quad_cls"]["recall"] if metrics.get("quad_cls") is not None else 0.0,
    }


def task_score_for_early_stop(task: str, score: Dict[str, float]) -> float:
    task_set = {x.strip().lower() for x in task.split(",") if x.strip()}
    if {"quad", "quadprediction", "asqp", "quadra"} & task_set:
        return score["quad_f1"]
    if {"quad_cls", "quadclass", "quad_classification"} & task_set:
        return score["quad_cls_f1"]
    if "mabsc" in task_set:
        return score["mabsc_span_f1"]
    if "macsa" in task_set:
        return score["macsa_span_f1"]
    values: List[float] = []
    if "mate" in task_set:
        values.append(score["mate_span_f1"])
    if "mote" in task_set:
        values.append(score["mote_span_f1"])
    if "macc" in task_set:
        values.append(score["macc_macro_f1"])
    if "masc" in task_set:
        values.append(score["masc_macro_f1"])
    if "aope" in task_set or "maope" in task_set:
        values.append(score["aope_macro_f1"])
    if values:
        return sum(values) / len(values)
    return (
        score["mate_span_f1"]
        + score["mote_span_f1"]
        + score["macc_macro_f1"]
        + score["masc_macro_f1"]
        + score["aope_macro_f1"]
    ) / 5.0


def train_one_epoch(
    model: MABSABaselineModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_fn=None,
    epoch_idx: int = 1,
    task: str = "mate,mote,macc,masc",
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0

    for step_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False), start=1):
        batch = move_to_device(batch, device)
        forward_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "image": batch["image"],
            "mate_labels": batch["mate_labels"],
            "mote_labels": batch["mote_labels"],
            "categories": batch["categories"],
            "aspect_spans": batch["aspect_spans"],
            "opinion_spans": batch["opinion_spans"],
            "aope_relations": batch["aope_relations"],
            "sentiments": batch["sentiments"],
            "task": task,
        }
        if model_accepts_arg(model, "mabsc_labels"):
            forward_kwargs["mabsc_labels"] = batch["mabsc_labels"]
        if model_accepts_arg(model, "macsa_labels"):
            forward_kwargs["macsa_labels"] = batch["macsa_labels"]
        outputs = model(**forward_kwargs)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu().item())
        steps += 1

        if log_fn is not None and (step_idx % 100 == 0 or step_idx == len(dataloader)):
            log_fn(f"[train][epoch {epoch_idx}][iter {step_idx}] loss={running_loss / steps:.6f}")

    return running_loss / max(steps, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MABSA baseline model")
    parser.add_argument("--train_dir", type=str, default="formatted_data/train")
    parser.add_argument("--val_dir", type=str, default="formatted_data/val")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--vision_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--num_categories", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--mate_loss_weight", type=float, default=1.0)
    parser.add_argument("--mote_loss_weight", type=float, default=1.0)
    parser.add_argument("--macc_loss_weight", type=float, default=1.0)
    parser.add_argument("--masc_loss_weight", type=float, default=1.0)
    parser.add_argument("--aope_loss_weight", type=float, default=1.0)
    parser.add_argument("--task", type=str, default="mate,mote,macc,masc")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    SAVE_DIR = args.save_dir
    LOG_DIR = args.log_dir
    TEXT_MODEL_NAME = args.text_model_name
    VISION_MODEL_NAME = args.vision_model_name
    NUM_CATEGORIES = args.num_categories
    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.epochs
    DROPOUT_P = args.dropout_p
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    MATE_LOSS_WEIGHT = args.mate_loss_weight
    MOTE_LOSS_WEIGHT = args.mote_loss_weight
    MACC_LOSS_WEIGHT = args.macc_loss_weight
    MASC_LOSS_WEIGHT = args.masc_loss_weight
    aope_loss_weight = args.aope_loss_weight
    TASK = args.task
    DEVICE = args.device

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{run_time}_{EPOCHS}_{BATCH_SIZE}_{LR}.log")

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    train_ds = PTDataset(TRAIN_DIR)
    val_ds = PTDataset(VAL_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = MABSABaselineModel(
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        num_categories=NUM_CATEGORIES,
        mate_loss_weight=MATE_LOSS_WEIGHT,
        mote_loss_weight=MOTE_LOSS_WEIGHT,
        macc_loss_weight=MACC_LOSS_WEIGHT,
        masc_loss_weight=MASC_LOSS_WEIGHT,
        aope_loss_weight=aope_loss_weight,
        dropout_p=DROPOUT_P,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = -1.0
    log(f"Log file: {log_path}")
    log(f"Device: {DEVICE}")
    log("Config:")
    log(f"  TRAIN_DIR={TRAIN_DIR}")
    log(f"  VAL_DIR={VAL_DIR}")
    log(f"  SAVE_DIR={SAVE_DIR}")
    log(f"  LOG_DIR={LOG_DIR}")
    log(f"  TEXT_MODEL_NAME={TEXT_MODEL_NAME}")
    log(f"  VISION_MODEL_NAME={VISION_MODEL_NAME}")
    log(f"  NUM_CATEGORIES={NUM_CATEGORIES}")
    log(f"  EPOCHS={EPOCHS}")
    log(f"  BATCH_SIZE={BATCH_SIZE}")
    log(f"  LR={LR}")
    log(f"  WEIGHT_DECAY={WEIGHT_DECAY}")
    log(f"  DROPOUT_P={DROPOUT_P}")
    log(f"  EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE}")
    log(f"  MATE_LOSS_WEIGHT={MATE_LOSS_WEIGHT}")
    log(f"  MOTE_LOSS_WEIGHT={MOTE_LOSS_WEIGHT}")
    log(f"  MACC_LOSS_WEIGHT={MACC_LOSS_WEIGHT}")
    log(f"  MASC_LOSS_WEIGHT={MASC_LOSS_WEIGHT}")
    log(f"  aope_loss_weight={aope_loss_weight}")
    log(f"  TASK={TASK}")
    log(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    no_improve_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            log_fn=log,
            epoch_idx=epoch,
            task=TASK,
        )
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        train_score = get_score_summary(train_metrics)
        val_score = get_score_summary(val_metrics)
        mean_score = task_score_for_early_stop(TASK, val_score)

        log(f"\nEpoch {epoch}/{EPOCHS}")
        log(f"train_loss: {train_loss:.6f}")
        log(
            f"train_metric - MATE_SPAN_F1: {train_score['mate_span_f1']:.4f} | "
            f"MOTE_SPAN_F1: {train_score['mote_span_f1']:.4f} | "
            f"MACC_MACRO_F1: {train_score['macc_macro_f1']:.4f} | "
            f"MASC_ACCURACY: {train_score['masc_accuracy']:.4f} | "
            f"MASC_MICRO_F1: {train_score['masc_micro_f1']:.4f} | "
            f"MASC_COUNT: {int(train_score['masc_correct'])}/{int(train_score['masc_total'])} | "
            f"MASC_MACRO_F1: {train_score['masc_macro_f1']:.4f} | "
            f"AOPE_F1: {train_score['aope_macro_f1']:.4f}"
        )
        log(
            f"val_metric   - MATE_SPAN_F1: {val_score['mate_span_f1']:.4f} | "
            f"MOTE_SPAN_F1: {val_score['mote_span_f1']:.4f} | "
            f"MACC_MACRO_F1: {val_score['macc_macro_f1']:.4f} | "
            f"MASC_ACCURACY: {val_score['masc_accuracy']:.4f} | "
            f"MASC_MICRO_F1: {val_score['masc_micro_f1']:.4f} | "
            f"MASC_COUNT: {int(val_score['masc_correct'])}/{int(val_score['masc_total'])} | "
            f"MASC_MACRO_F1: {val_score['masc_macro_f1']:.4f} | "
            f"AOPE_F1: {val_score['aope_macro_f1']:.4f}"
        )

        # log("\n[VAL][MATE] classification report")
        # log(val_metrics["mate"]["classification_report"])
        # log(format_confusion_matrix(val_metrics["mate"]["confusion_matrix"], val_metrics["mate"]["label_names"]))

        # log("\n[VAL][MOTE] classification report")
        # log(val_metrics["mote"]["classification_report"])
        # log(format_confusion_matrix(val_metrics["mote"]["confusion_matrix"], val_metrics["mote"]["label_names"]))

        # if val_metrics["masc"] is not None:
        #     log("\n[VAL][MASC] classification report")
        #     log(val_metrics["masc"]["classification_report"])
        #     log(
        #         format_confusion_matrix(
        #             val_metrics["masc"]["confusion_matrix"],
        #             val_metrics["masc"]["label_names"],
        #         )
        #     )

        # log("\n[VAL][MACC] classification report")
        # log(val_metrics["macc"]["classification_report"])
        # log(format_confusion_matrix(val_metrics["macc"]["confusion_matrix"], val_metrics["macc"]["label_names"]))

        if mean_score > best_val:
            best_val = mean_score
            no_improve_epochs = 0
            ckpt_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            log(f"Saved best model to: {ckpt_path}")
        else:
            no_improve_epochs += 1
            log(f"No improvement for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                log(f"Early stopping triggered at epoch {epoch}.")
                break



