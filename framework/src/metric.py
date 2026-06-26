from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _to_1d_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    return arr.reshape(-1)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    label_to_idx = {int(label): idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        if int(true_label) in label_to_idx and int(pred_label) in label_to_idx:
            cm[label_to_idx[int(true_label)], label_to_idx[int(pred_label)]] += 1
    return cm


def _prf_from_confusion(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    precision = []
    recall = []
    f1 = []
    support = []
    for idx in range(cm.shape[0]):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)
        support.append(int(cm[idx, :].sum()))
    return (
        np.asarray(precision, dtype=float),
        np.asarray(recall, dtype=float),
        np.asarray(f1, dtype=float),
        np.asarray(support, dtype=int),
    )


def _classification_report(
    names: Sequence[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
) -> str:
    total = int(support.sum())
    macro_precision = float(np.mean(precision)) if len(precision) else 0.0
    macro_recall = float(np.mean(recall)) if len(recall) else 0.0
    weighted_precision = float(np.average(precision, weights=support)) if total > 0 else 0.0
    weighted_recall = float(np.average(recall, weights=support)) if total > 0 else 0.0

    lines = [f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>9}", ""]
    for name, p, r, score, sup in zip(names, precision, recall, f1, support):
        lines.append(f"{name:>12} {p:>10.4f} {r:>10.4f} {score:>10.4f} {int(sup):>9}")
    lines.append("")
    lines.append(f"{'accuracy':>12} {'':>10} {'':>10} {accuracy:>10.4f} {total:>9}")
    lines.append(f"{'macro avg':>12} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f} {total:>9}")
    lines.append(
        f"{'weighted avg':>12} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {total:>9}"
    )
    return "\n".join(lines)


def evaluate_classification(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    label_names: Optional[Sequence[str]] = None,
    ignore_label: Optional[int] = None,
) -> Dict[str, object]:
    """
    Evaluate single-label classification with F1 and confusion matrix details.
    """
    y_true_arr = _to_1d_array(y_true)
    y_pred_arr = _to_1d_array(y_pred)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")

    if ignore_label is not None:
        mask = y_true_arr != ignore_label
        y_true_arr = y_true_arr[mask]
        y_pred_arr = y_pred_arr[mask]

    labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    if not labels:
        raise ValueError("No valid labels after filtering.")

    cm = _confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    p, r, f1, s = _prf_from_confusion(cm)

    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    correct = int((y_true_arr == y_pred_arr).sum())
    total = int(y_true_arr.size)
    accuracy = float(correct / total) if total > 0 else 0.0
    micro_f1 = accuracy
    weighted_f1 = float(np.average(f1, weights=s)) if total > 0 else 0.0

    names: List[str] = []
    for idx, lab in enumerate(labels):
        if label_names is not None and lab < len(label_names):
            names.append(label_names[lab])
        else:
            names.append(str(lab))

    per_class = []
    iou_values: List[float] = []
    for i, lab in enumerate(labels):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        iou_values.append(iou)
        per_class.append(
            {
                "label_id": int(lab),
                "label_name": names[i],
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "iou": float(iou),
                "support": int(s[i]),
            }
        )
    macro_iou = float(np.mean(iou_values)) if iou_values else 0.0

    report = _classification_report(names, p, r, f1, s, accuracy, macro_f1, weighted_f1)

    return {
        "labels": labels,
        "label_names": names,
        "macro_f1": float(macro_f1),
        "macro_iou": macro_iou,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def evaluate_multilabel_f1(
    y_true: Sequence[Sequence[int]] | np.ndarray,
    y_pred: Sequence[Sequence[int]] | np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate multi-label classification (e.g., ACC) from binary indicator matrix.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have same shape for multilabel.")

    tp = (y_true_arr * y_pred_arr).sum(axis=0).astype(float)
    fp = ((1 - y_true_arr) * y_pred_arr).sum(axis=0).astype(float)
    fn = (y_true_arr * (1 - y_pred_arr)).sum(axis=0).astype(float)
    per_label_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    support = y_true_arr.sum(axis=0).astype(float)
    micro_tp = float(tp.sum())
    micro_fp = float(fp.sum())
    micro_fn = float(fn.sum())
    micro_f1 = 2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn) if (2 * micro_tp + micro_fp + micro_fn) > 0 else 0.0
    sample_tp = (y_true_arr * y_pred_arr).sum(axis=1).astype(float)
    sample_fp = ((1 - y_true_arr) * y_pred_arr).sum(axis=1).astype(float)
    sample_fn = (y_true_arr * (1 - y_pred_arr)).sum(axis=1).astype(float)
    sample_den = 2 * sample_tp + sample_fp + sample_fn
    sample_f1 = np.divide(2 * sample_tp, sample_den, out=np.zeros_like(sample_tp, dtype=float), where=sample_den > 0)
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(per_label_f1)) if len(per_label_f1) else 0.0,
        "weighted_f1": float(np.average(per_label_f1, weights=support)) if support.sum() > 0 else 0.0,
        "samples_f1": float(np.mean(sample_f1)) if len(sample_f1) else 0.0,
    }


def bio_tags_to_spans(
    tags: Sequence[int] | np.ndarray,
    attention_mask: Optional[Sequence[int] | np.ndarray] = None,
    b_label: int = 1,
    i_label: int = 2,
    ignore_label: int = -100,
) -> List[Tuple[int, int]]:
    tag_list = np.asarray(tags).reshape(-1).tolist()
    mask_list = (
        np.asarray(attention_mask).reshape(-1).tolist()
        if attention_mask is not None
        else [1] * len(tag_list)
    )
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, (tag, keep) in enumerate(zip(tag_list, mask_list)):
        if not keep or tag == ignore_label:
            if start is not None:
                spans.append((start, idx))
                start = None
            continue
        if tag == b_label:
            if start is not None:
                spans.append((start, idx))
            start = idx
        elif tag == i_label:
            if start is None:
                start = idx
        else:
            if start is not None:
                spans.append((start, idx))
                start = None
    if start is not None:
        spans.append((start, len(tag_list)))
    return spans


def evaluate_span_sentiment_sets(
    y_true: Sequence[Sequence[Tuple[Tuple[int, int], int]]],
    y_pred: Sequence[Sequence[Tuple[Tuple[int, int], int]]],
) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same number of samples.")

    tp = 0
    fp = 0
    fn = 0
    only_fp = 0
    only_fn = 0
    fp_fn = 0
    exact_match = 0
    for true_items, pred_items in zip(y_true, y_pred):
        true_set = set(true_items)
        pred_set = set(pred_items)
        sample_tp = len(true_set & pred_set)
        sample_fp = len(pred_set - true_set)
        sample_fn = len(true_set - pred_set)
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
        if sample_fp == 0 and sample_fn == 0:
            exact_match += 1
        elif sample_fp > 0 and sample_fn > 0:
            fp_fn += 1
        elif sample_fp > 0:
            only_fp += 1
        elif sample_fn > 0:
            only_fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "only_fp": float(only_fp),
        "only_fn": float(only_fn),
        "fp_fn": float(fp_fn),
        "exact_match": float(exact_match),
        "num_samples": float(len(y_true)),
    }


def evaluate_relation_sets(
    y_true: Sequence[Sequence[Tuple[int, int]]],
    y_pred: Sequence[Sequence[Tuple[int, int]]],
) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same number of samples.")

    tp = 0
    fp = 0
    fn = 0
    for true_items, pred_items in zip(y_true, y_pred):
        true_set = set(true_items)
        pred_set = set(pred_items)
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate_quad_sets(
    y_true: Sequence[Sequence[Tuple[Tuple[int, int], int, Tuple[int, int], int]]],
    y_pred: Sequence[Sequence[Tuple[Tuple[int, int], int, Tuple[int, int], int]]],
) -> Dict[str, float]:
    base = evaluate_relation_sets(y_true, y_pred)

    def project_item(item: Tuple[Tuple[int, int], int, Tuple[int, int], int], name: str) -> Tuple[object, ...]:
        aspect_span, category, opinion_span, sentiment = item
        if name == "aspect":
            return (aspect_span,)
        if name == "opinion":
            return (opinion_span,)
        if name == "category":
            return (category,)
        if name == "sentiment":
            return (sentiment,)
        if name == "aspect_opinion":
            return (aspect_span, opinion_span)
        if name == "aspect_category":
            return (aspect_span, category)
        if name == "aspect_sentiment":
            return (aspect_span, sentiment)
        if name == "opinion_sentiment":
            return (opinion_span, sentiment)
        if name == "aspect_opinion_category":
            return (aspect_span, category, opinion_span)
        if name == "aspect_opinion_sentiment":
            return (aspect_span, opinion_span, sentiment)
        if name == "aspect_category_sentiment":
            return (aspect_span, category, sentiment)
        return item

    component_names = [
        "aspect",
        "opinion",
        "category",
        "sentiment",
        "aspect_opinion",
        "aspect_category",
        "aspect_sentiment",
        "opinion_sentiment",
        "aspect_opinion_category",
        "aspect_opinion_sentiment",
        "aspect_category_sentiment",
    ]
    components: Dict[str, Dict[str, float]] = {}
    for name in component_names:
        true_projected = [[project_item(item, name) for item in items] for items in y_true]
        pred_projected = [[project_item(item, name) for item in items] for items in y_pred]
        components[name] = evaluate_relation_sets(true_projected, pred_projected)

    base["components"] = components
    base["category_sentiment"] = evaluate_quad_category_sentiment_sets(y_true, y_pred)
    return base


def evaluate_quad_category_sentiment_sets(
    y_true: Sequence[Sequence[Tuple[Tuple[int, int], int, Tuple[int, int], int]]],
    y_pred: Sequence[Sequence[Tuple[Tuple[int, int], int, Tuple[int, int], int]]],
) -> Dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same number of samples.")

    category_names = ["Facility", "Service", "Amenity", "Experience", "Branding", "Loyalty"]
    sentiment_names = ["Negative", "Neutral", "Positive"]
    per_class = []

    for sentiment_id, sentiment_name in enumerate(sentiment_names):
        for category_id, category_name in enumerate(category_names):
            tp = 0
            fp = 0
            fn = 0
            for true_items, pred_items in zip(y_true, y_pred):
                true_set = set(true_items)
                pred_set = set(pred_items)

                true_class = {
                    item
                    for item in true_set
                    if int(item[1]) == category_id and int(item[3]) == sentiment_id
                }
                pred_class = {
                    item
                    for item in pred_set
                    if int(item[1]) == category_id and int(item[3]) == sentiment_id
                }

                tp += len(true_class & pred_class)
                fp += len(pred_class - true_class)
                fn += len(true_class - pred_class)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class.append(
                {
                    "category_id": category_id,
                    "sentiment_id": sentiment_id,
                    "category": category_name,
                    "sentiment": sentiment_name,
                    "label": f"{sentiment_name} {category_name}",
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "tp": float(tp),
                    "fp": float(fp),
                    "fn": float(fn),
                    "support": float(tp + fn),
                }
            )

    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0
    macro_precision = float(np.mean([row["precision"] for row in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([row["recall"] for row in per_class])) if per_class else 0.0
    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def format_quad_category_sentiment_report(metrics: Dict[str, object]) -> str:
    rows = metrics.get("per_class", []) if metrics is not None else []
    lines = ["label".ljust(28) + "precision  recall     f1        tp     fp     fn     support"]
    for row in rows:
        support = int(row["support"])
        if support == 0 and int(row["fp"]) == 0:
            continue
        lines.append(
            f"{str(row['label']).ljust(28)}"
            f"{row['precision']:<10.4f}"
            f"{row['recall']:<10.4f}"
            f"{row['f1']:<10.4f}"
            f"{int(row['tp']):<7}"
            f"{int(row['fp']):<7}"
            f"{int(row['fn']):<7}"
            f"{support:<7}"
        )
    lines.append(
        f"{'macro avg'.ljust(28)}"
        f"{metrics.get('macro_precision', 0.0):<10.4f}"
        f"{metrics.get('macro_recall', 0.0):<10.4f}"
        f"{metrics.get('macro_f1', 0.0):<10.4f}"
    )
    return "\n".join(lines)


def mabsc_tags_to_span_sentiments(
    tags: Sequence[int] | np.ndarray,
    attention_mask: Optional[Sequence[int] | np.ndarray] = None,
    ignore_label: int = -100,
) -> List[Tuple[Tuple[int, int], int]]:
    tag_list = np.asarray(tags).reshape(-1).tolist()
    mask_list = (
        np.asarray(attention_mask).reshape(-1).tolist()
        if attention_mask is not None
        else [1] * len(tag_list)
    )
    items: List[Tuple[Tuple[int, int], int]] = []
    start: Optional[int] = None
    sentiment: Optional[int] = None
    for idx, (tag, keep) in enumerate(zip(tag_list, mask_list)):
        if not keep or tag == ignore_label:
            if start is not None and sentiment is not None:
                items.append(((start, idx), sentiment))
            start = None
            sentiment = None
            continue
        if tag in {1, 2, 3}:
            if start is not None and sentiment is not None:
                items.append(((start, idx), sentiment))
            start = idx
            sentiment = int(tag) - 1
        elif tag == 4:
            if start is None or sentiment is None:
                if start is not None and sentiment is not None:
                    items.append(((start, idx), sentiment))
                start = idx
        else:
            if start is not None and sentiment is not None:
                items.append(((start, idx), sentiment))
            start = None
            sentiment = None
    if start is not None and sentiment is not None:
        items.append(((start, len(tag_list)), sentiment))
    return items


def macsa_tags_to_span_categories(
    tags: Sequence[int] | np.ndarray,
    attention_mask: Optional[Sequence[int] | np.ndarray] = None,
    ignore_label: int = -100,
) -> List[Tuple[Tuple[int, int], int]]:
    tag_list = np.asarray(tags).reshape(-1).tolist()
    mask_list = (
        np.asarray(attention_mask).reshape(-1).tolist()
        if attention_mask is not None
        else [1] * len(tag_list)
    )
    items: List[Tuple[Tuple[int, int], int]] = []
    start: Optional[int] = None
    category: Optional[int] = None
    for idx, (tag, keep) in enumerate(zip(tag_list, mask_list)):
        if not keep or tag == ignore_label:
            if start is not None and category is not None:
                items.append(((start, idx), category))
            start = None
            category = None
            continue
        if tag in {1, 2, 3, 4, 5, 6}:
            if start is not None and category is not None:
                items.append(((start, idx), category))
            start = idx
            category = int(tag) - 1
        elif tag == 7:
            if start is None or category is None:
                start = idx
        else:
            if start is not None and category is not None:
                items.append(((start, idx), category))
            start = None
            category = None
    if start is not None and category is not None:
        items.append(((start, len(tag_list)), category))
    return items


def format_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
) -> str:
    """
    Pretty text table for confusion matrix.
    """
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    if cm.shape[0] != len(labels):
        raise ValueError("labels length must match confusion matrix size.")

    width = max(7, max(len(x) for x in labels) + 2)
    header = "true\\pred".ljust(width) + "".join(x.ljust(width) for x in labels)
    lines = [header]

    for i, row_name in enumerate(labels):
        row = row_name.ljust(width) + "".join(str(int(v)).ljust(width) for v in cm[i])
        lines.append(row)

    return "\n".join(lines)
