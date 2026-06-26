from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List


TIMESTAMP_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*")
ROW_RE = re.compile(
    r"^(?P<label>(Negative|Neutral|Positive)\s+\S+)\s+"
    r"(?P<precision>\d+\.\d+)\s+"
    r"(?P<recall>\d+\.\d+)\s+"
    r"(?P<f1>\d+\.\d+)\s+"
    r"(?P<tp>\d+)\s+"
    r"(?P<fp>\d+)\s+"
    r"(?P<fn>\d+)\s+"
    r"(?P<support>\d+)"
)


def latest_log(log_dir: Path) -> Path | None:
    files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def parse_report(log_path: Path, model_name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    in_section = False
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = TIMESTAMP_RE.sub("", raw_line.strip())
            if "category+sentiment P/R/F1" in line:
                in_section = True
                continue
            if in_section and line.startswith("="):
                if rows:
                    break
                continue
            if not in_section:
                continue
            match = ROW_RE.match(line)
            if match:
                item = match.groupdict()
                rows.append(
                    {
                        "model": model_name,
                        "label": item["label"],
                        "precision": item["precision"],
                        "recall": item["recall"],
                        "f1": item["f1"],
                        "tp": item["tp"],
                        "fp": item["fp"],
                        "fn": item["fn"],
                        "support": item["support"],
                        "log_file": str(log_path),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Quad category+sentiment P/R/F1 rows from eval logs.")
    parser.add_argument("--log_root", type=str, default="final_log/quad_category_sentiment_four_models")
    parser.add_argument("--output_csv", type=str, default="results/quad_category_sentiment_four_models.csv")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    for model_dir in sorted([p for p in log_root.iterdir() if p.is_dir()]) if log_root.exists() else []:
        log_path = latest_log(model_dir)
        if log_path is None:
            print(f"[SKIP] no log file: {model_dir}")
            continue
        rows = parse_report(log_path, model_dir.name)
        print(f"[COLLECT] {model_dir.name}: {len(rows)} rows from {log_path}")
        all_rows.extend(rows)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "label", "precision", "recall", "f1", "tp", "fp", "fn", "support", "log_file"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
