from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import json

from PIL import Image


MATE_TAGS = {
    "O": 0,
    "B-ASP": 1,
    "I-ASP": 2,
}

MOTE_TAGS = {
    "O": 0,
    "B-OPN": 1,
    "I-OPN": 2,
}

SENTIMENT_MAP = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

# ALL_CATEGORIES = [
#     "Facility",
#     "Service",
#     "Amenity",
#     "Experience",
#     "Branding",
#     "Loyalty",
# ]

ALL_CATEGORIES = [
    "FACILITY",
    "SERVICE",
    "AMENITY",
    "EXPERIENCE",
    "BRANDING",
    "LOYALTY",
]

Span = Tuple[int, int]


def _token_word_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    in_token = False
    start = 0
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_token:
                spans.append((start, i))
                in_token = False
        else:
            if not in_token:
                start = i
                in_token = True
    if in_token:
        spans.append((start, len(text)))
    return spans


def _word_span_to_char_span(text: str, word_start: int, word_end: int) -> Optional[Span]:
    if word_start < 0 or word_end < 0 or word_end <= word_start:
        return None
    spans = _token_word_spans(text)
    if word_start >= len(spans) or word_end > len(spans):
        return None
    char_start = spans[word_start][0]
    char_end = spans[word_end - 1][1]
    return char_start, char_end


def _char_to_token_span(
    offsets: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Optional[Span]:
    token_ids: List[int] = []
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:
            continue
        if tok_start < char_end and tok_end > char_start:
            token_ids.append(i)
    if not token_ids:
        return None
    return token_ids[0], token_ids[-1] + 1


def _load_image(
    sample: Dict[str, Any],
    image_root: Optional[str | Path],
    image_processor: Optional[Any],
) -> Tuple[Optional[Any], bool]:
    photo_name = sample.get("review_photo")
    if not photo_name or image_root is None:
        return None, False

    photo_name = str(photo_name).strip()
    image: Optional[Image.Image] = None

    if photo_name.startswith("http://") or photo_name.startswith("https://"):
        cache_dir = Path(image_root)
        cache_dir.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(photo_name)
        cache_name = Path(parsed.path).name or "downloaded_image.jpg"
        cache_path = cache_dir / cache_name

        if cache_path.exists():
            try:
                image = Image.open(cache_path).convert("RGB")
            except Exception:
                image = None
        else:
            try:
                req = Request(photo_name, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=15) as resp:
                    image_bytes = resp.read()
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                try:
                    image.save(cache_path)
                except Exception:
                    pass
            except HTTPError as e:
                if e.code == 404:
                    return None, True
                return None, False
            except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                return None, False
    else:
        image_path = Path(image_root) / photo_name
        if not image_path.exists():
            return None, False
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return None, False

    if image is None:
        return None, False

    if image_processor is None:
        return image, False

    processed = image_processor(image)
    if isinstance(processed, dict) and "pixel_values" in processed:
        pixel_values = processed["pixel_values"]
        if isinstance(pixel_values, list) and pixel_values:
            return pixel_values[0], False
        return pixel_values, False
    return processed, False


def preprocess_sample(
    sample: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
    image_root: Optional[str | Path] = None,
    image_processor: Optional[Any] = None,
    ignore_empty_span_samples: bool = True,
) -> Optional[Dict[str, Any]]:
    text = str(sample.get("review", ""))
    rows = sample.get("extraction", [])
    if ignore_empty_span_samples and not rows:
        return None

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]
    seq_len = len(input_ids)

    mate_labels = [MATE_TAGS["O"]] * seq_len
    mote_labels = [MOTE_TAGS["O"]] * seq_len
    aspect_spans: List[Span] = []
    opinion_spans: List[Span] = []
    sentiments: List[int] = []
    categories: List[int] = []
    aope_relations: List[Tuple[int, int]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        polarity = str(row.get("Polarity", "")).strip()
        category = str(row.get("Category", "")).strip()
        if polarity not in SENTIMENT_MAP:
            continue

        asp_span_words = row.get("Aspect_span", [-1, -1])
        opn_span_words = row.get("Opinion_span", [-1, -1])
        if not isinstance(asp_span_words, list) or not isinstance(opn_span_words, list):
            continue
        if len(asp_span_words) != 2 or len(opn_span_words) != 2:
            continue

        asp_char_span = _word_span_to_char_span(text, int(asp_span_words[0]), int(asp_span_words[1]))
        opn_char_span = _word_span_to_char_span(text, int(opn_span_words[0]), int(opn_span_words[1]))
        if asp_char_span is None or opn_char_span is None:
            continue

        asp_tok_span = _char_to_token_span(offsets, asp_char_span[0], asp_char_span[1])
        opn_tok_span = _char_to_token_span(offsets, opn_char_span[0], opn_char_span[1])
        if asp_tok_span is None or opn_tok_span is None:
            continue

        a_start, a_end = asp_tok_span
        o_start, o_end = opn_tok_span

        mate_labels[a_start] = MATE_TAGS["B-ASP"]
        for i in range(a_start + 1, a_end):
            mate_labels[i] = MATE_TAGS["I-ASP"]

        mote_labels[o_start] = MOTE_TAGS["B-OPN"]
        for i in range(o_start + 1, o_end):
            mote_labels[i] = MOTE_TAGS["I-OPN"]

        aspect_spans.append((a_start, a_end))
        opinion_spans.append((o_start, o_end))
        sentiments.append(SENTIMENT_MAP[polarity])
        categories.append(ALL_CATEGORIES.index(category) if category in ALL_CATEGORIES else -100)
        aope_relations.append((len(aspect_spans) - 1, len(opinion_spans) - 1))

    if ignore_empty_span_samples and not aspect_spans:
        return None

    image, image_404 = _load_image(sample, image_root=image_root, image_processor=image_processor)
    if image_404:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mate_labels": mate_labels,
        "mote_labels": mote_labels,
        "aspect_spans": aspect_spans,
        "opinion_spans": opinion_spans,
        "aope_relations": aope_relations,
        "sentiments": sentiments,
        "categories": categories,
        "image": image,
    }


def preprocess_dataset(
    samples: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    image_root: Optional[str | Path] = None,
    image_processor: Optional[Any] = None,
    ignore_empty_span_samples: bool = True,
) -> List[Dict[str, Any]]:
    from tqdm import tqdm

    outputs: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="Preprocessing samples"):
        result = preprocess_sample(
            sample=sample,
            tokenizer=tokenizer,
            max_length=max_length,
            image_root=image_root,
            image_processor=image_processor,
            ignore_empty_span_samples=ignore_empty_span_samples,
        )
        if result is not None:
            outputs.append(result)
    return outputs


if __name__ == "__main__":
    import argparse
    import os
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoImageProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, choices=["train", "val", "test"], required=True)
    parser.add_argument("--mode", type=str, choices=["text", "image", "both"], required=True)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="formatted_data")
    parser.add_argument("--text_encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--image_encoder_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--image_root", type=str, default="images")
    parser.add_argument("--ignore_empty_span_samples", action="store_true")
    args = parser.parse_args()

    def encoder_dir_name(name: str) -> str:
        return name.replace("/", "__")

    input_path = args.input_path or f"data/final_samples_span_{args.set_type}.json"
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.mode in {"text", "both"}:
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None

    if args.mode in {"image", "both"}:
        image_processor = AutoImageProcessor.from_pretrained(args.image_encoder_name)
    else:
        image_processor = None

    if args.mode == "text":
        out_dir = Path(args.output_root) / encoder_dir_name(args.text_encoder_name) / args.set_type
    elif args.mode == "image":
        out_dir = Path(args.output_root) / encoder_dir_name(args.image_encoder_name) / args.set_type
    else:
        out_dir = Path(args.output_root) / args.set_type

    os.makedirs(out_dir, exist_ok=True)

    for sample in tqdm(samples, desc=f"Saving .pt files ({args.mode})"):
        out_name = str(sample.get("json_file", "")).strip()
        if not out_name:
            continue

        if args.mode == "image":
            image, image_404 = _load_image(
                sample=sample,
                image_root=args.image_root,
                image_processor=lambda img: image_processor(img, return_tensors="pt")["pixel_values"][0],
            )
            if image_404:
                continue
            torch.save({"image": image}, out_dir / out_name.replace(".json", ".pt"))
            continue

        output = preprocess_sample(
            sample=sample,
            tokenizer=tokenizer,
            max_length=args.max_length,
            image_root=args.image_root if args.mode == "both" else None,
            image_processor=(
                (lambda img: image_processor(img, return_tensors="pt")["pixel_values"][0])
                if args.mode == "both"
                else None
            ),
            ignore_empty_span_samples=args.ignore_empty_span_samples,
        )
        if output is None:
            continue

        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output["mate_labels"] = torch.tensor(output["mate_labels"], dtype=torch.long)
        output["mote_labels"] = torch.tensor(output["mote_labels"], dtype=torch.long)

        torch.save(output, out_dir / out_name.replace(".json", ".pt"))
