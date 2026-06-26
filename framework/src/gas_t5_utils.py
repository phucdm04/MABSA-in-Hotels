from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from metric import evaluate_classification, evaluate_quad_sets, evaluate_span_sentiment_sets


CATEGORY_TO_ID = {
    "FACILITY": 0,
    "SERVICE": 1,
    "AMENITY": 2,
    "EXPERIENCE": 3,
    "BRANDING": 4,
    "LOYALTY": 5,
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}
CATEGORY_ALIASES = {
    "FAC": "FACILITY",
    "SER": "SERVICE",
    "AME": "AMENITY",
    "EXP": "EXPERIENCE",
    "BRA": "BRANDING",
    "LOY": "LOYALTY",
}

SENTIMENT_TO_ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID_TO_SENTIMENT = {v: k for k, v in SENTIMENT_TO_ID.items()}
SENTIMENT_ALIASES = {
    "NEG": "NEGATIVE",
    "NEU": "NEUTRAL",
    "POS": "POSITIVE",
}


@dataclass
class GasExample:
    source: str
    target: str
    review: str
    gold: Any
    json_file: str = ""


def normalize_label(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def normalize_category(value: Any) -> Optional[str]:
    label = normalize_label(value)
    label = CATEGORY_ALIASES.get(label, label)
    return label if label in CATEGORY_TO_ID else None


def normalize_sentiment(value: Any) -> Optional[str]:
    label = normalize_label(value)
    label = SENTIMENT_ALIASES.get(label, label)
    return label if label in SENTIMENT_TO_ID else None


def get_extractions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    ext = sample.get("extraction", [])
    return ext if isinstance(ext, list) else []


def get_review(sample: Dict[str, Any]) -> str:
    return str(sample.get("review", sample.get("text", sample.get("sentence", "")))).strip()


def get_json_file(sample: Dict[str, Any]) -> str:
    return str(sample.get("json_file", sample.get("id", "")))


def get_aspect_term(item: Dict[str, Any]) -> str:
    return str(item.get("aspect_term", item.get("aspect", ""))).strip()


def get_opinion_term(item: Dict[str, Any]) -> str:
    return str(item.get("opinion_term", item.get("opinion", ""))).strip()


def get_category(item: Dict[str, Any]) -> Optional[str]:
    return normalize_category(item.get("Category", item.get("aspect_category", item.get("category", ""))))


def get_sentiment(item: Dict[str, Any]) -> Optional[str]:
    return normalize_sentiment(item.get("Polarity", item.get("sentiment", item.get("polarity", ""))))


def format_items(items: Iterable[Sequence[str]]) -> str:
    rows = []
    seen = set()
    for item in items:
        normalized = tuple(str(x).strip() for x in item)
        if not all(normalized) or normalized in seen:
            continue
        seen.add(normalized)
        rows.append(" | ".join(normalized))
    return " ; ".join(rows) if rows else "none"


def build_gas_examples(samples: Sequence[Dict[str, Any]], task: str) -> List[GasExample]:
    task = task.lower()
    examples: List[GasExample] = []
    for sample in samples:
        review = get_review(sample)
        if not review:
            continue
        json_file = get_json_file(sample)
        extractions = get_extractions(sample)
        if task == "macc":
            for item in extractions:
                aspect = get_aspect_term(item)
                category = get_category(item)
                if not aspect or category is None:
                    continue
                source = f"macc: review: {review} aspect: {aspect}"
                examples.append(GasExample(source, category, review, (aspect, CATEGORY_TO_ID[category]), json_file))
        elif task == "macsa":
            pairs = []
            gold = []
            for item in extractions:
                aspect = get_aspect_term(item)
                category = get_category(item)
                if not aspect or category is None:
                    continue
                pairs.append((aspect, category))
                gold.append((aspect, CATEGORY_TO_ID[category]))
            source = f"macsa: review: {review}"
            examples.append(GasExample(source, format_items(pairs), review, gold, json_file))
        elif task in {"quad", "quadra", "asqp"}:
            quads = []
            gold = []
            for item in extractions:
                aspect = get_aspect_term(item)
                opinion = get_opinion_term(item)
                category = get_category(item)
                sentiment = get_sentiment(item)
                if not aspect or not opinion or category is None or sentiment is None:
                    continue
                quads.append((aspect, category, opinion, sentiment))
                gold.append((aspect, CATEGORY_TO_ID[category], opinion, SENTIMENT_TO_ID[sentiment]))
            source = f"quad: review: {review}"
            examples.append(GasExample(source, format_items(quads), review, gold, json_file))
        else:
            raise ValueError(f"Unsupported GAS task: {task}")
    return examples


def split_generated_items(text: str) -> List[List[str]]:
    text = str(text or "").strip()
    if not text or text.lower() in {"none", "null", "[]"}:
        return []
    rows = re.split(r"\s*;\s*", text)
    items = []
    for row in rows:
        parts = [p.strip() for p in row.split("|")]
        if any(parts):
            items.append(parts)
    return items


def simple_word_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def char_to_word_span(text: str, start_char: int, end_char: int) -> Optional[Tuple[int, int]]:
    words = simple_word_spans(text)
    covered = [idx for idx, (_, s, e) in enumerate(words) if s < end_char and e > start_char]
    if not covered:
        return None
    return (covered[0], covered[-1] + 1)


def find_term_span(text: str, term: str) -> Optional[Tuple[int, int]]:
    term = str(term or "").strip()
    if not term:
        return None
    pattern = re.escape(term)
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        compact_text = re.sub(r"\s+", " ", text)
        compact_match = re.search(pattern, compact_text, flags=re.IGNORECASE)
        if compact_match is None:
            return None
        match = compact_match
        text = compact_text
    return char_to_word_span(text, match.start(), match.end())


def parse_macc_prediction(text: str) -> Optional[int]:
    label = normalize_category(text)
    return CATEGORY_TO_ID[label] if label is not None else None


def parse_macsa_prediction(text: str, review: str) -> List[Tuple[Tuple[int, int], int]]:
    pred = []
    seen = set()
    for parts in split_generated_items(text):
        if len(parts) < 2:
            continue
        span = find_term_span(review, parts[0])
        category = normalize_category(parts[1])
        if span is None or category is None:
            continue
        item = (span, CATEGORY_TO_ID[category])
        if item not in seen:
            seen.add(item)
            pred.append(item)
    return pred


def parse_quad_prediction(text: str, review: str) -> List[Tuple[Tuple[int, int], int, Tuple[int, int], int]]:
    pred = []
    seen = set()
    for parts in split_generated_items(text):
        if len(parts) < 4:
            continue
        aspect_span = find_term_span(review, parts[0])
        category = normalize_category(parts[1])
        opinion_span = find_term_span(review, parts[2])
        sentiment = normalize_sentiment(parts[3])
        if aspect_span is None or opinion_span is None or category is None or sentiment is None:
            continue
        item = (aspect_span, CATEGORY_TO_ID[category], opinion_span, SENTIMENT_TO_ID[sentiment])
        if item not in seen:
            seen.add(item)
            pred.append(item)
    return pred


def gold_macsa_sets(examples: Sequence[GasExample]) -> List[List[Tuple[Tuple[int, int], int]]]:
    all_items = []
    for ex in examples:
        items = []
        for aspect, category_id in ex.gold:
            span = find_term_span(ex.review, aspect)
            if span is not None:
                items.append((span, int(category_id)))
        all_items.append(items)
    return all_items


def gold_quad_sets(examples: Sequence[GasExample]) -> List[List[Tuple[Tuple[int, int], int, Tuple[int, int], int]]]:
    all_items = []
    for ex in examples:
        items = []
        for aspect, category_id, opinion, sentiment_id in ex.gold:
            aspect_span = find_term_span(ex.review, aspect)
            opinion_span = find_term_span(ex.review, opinion)
            if aspect_span is not None and opinion_span is not None:
                items.append((aspect_span, int(category_id), opinion_span, int(sentiment_id)))
        all_items.append(items)
    return all_items


def evaluate_gas_outputs(task: str, examples: Sequence[GasExample], predictions: Sequence[str]) -> Dict[str, Any]:
    task = task.lower()
    if task == "macc":
        y_true = []
        y_pred = []
        for ex, pred_text in zip(examples, predictions):
            _, gold_category = ex.gold
            pred_category = parse_macc_prediction(pred_text)
            y_true.append(int(gold_category))
            y_pred.append(int(pred_category) if pred_category is not None else len(CATEGORY_TO_ID))
        return {
            "macc": evaluate_classification(
                y_true,
                y_pred,
                label_names=["Facility", "Service", "Amenity", "Experience", "Branding", "Loyalty", "Invalid"],
                ignore_label=-100,
            )
        }
    if task == "macsa":
        y_true = gold_macsa_sets(examples)
        y_pred = [parse_macsa_prediction(pred, ex.review) for ex, pred in zip(examples, predictions)]
        return {"macsa_span": evaluate_span_sentiment_sets(y_true, y_pred)}
    if task in {"quad", "quadra", "asqp"}:
        y_true = gold_quad_sets(examples)
        y_pred = [parse_quad_prediction(pred, ex.review) for ex, pred in zip(examples, predictions)]
        return {"quad": evaluate_quad_sets(y_true, y_pred)}
    raise ValueError(f"Unsupported GAS task: {task}")
