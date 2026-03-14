#!/usr/bin/env python3
"""Format recall-only metrics into route comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


ROUTES = [
    ("main", "route_main", "avg_main_candidates"),
    ("content", "route_content", "avg_content_candidates"),
    ("hstu", "route_hstu", "avg_hstu_candidates"),
    ("merged", "", "avg_recall_candidates"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format recall route metrics as markdown/csv tables.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/mnt/sda/hfx/KuaiRand/baseline/checkpoints/recall_only_compare/metrics.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <input-json stem>_table.md",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to <input-json stem>_table.csv",
    )
    return parser.parse_args()


def _infer_topks(metric_dict: Dict[str, float]) -> List[int]:
    out = set()
    for key in metric_dict:
        if key.startswith("hr@"):
            try:
                out.add(int(key.split("@", 1)[1]))
            except ValueError:
                continue
    return sorted(out)


def _metric_value(metric_dict: Dict[str, float], prefix: str, metric_name: str, k: int) -> float:
    key = f"{metric_name}@{k}" if not prefix else f"{prefix}_{metric_name}@{k}"
    return float(metric_dict.get(key, 0.0))


def _format_float(x: float) -> str:
    return f"{float(x):.6f}"


def _route_rows(metric_dict: Dict[str, float], topks: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for route_name, prefix, avg_key in ROUTES:
        row: Dict[str, object] = {
            "route": route_name,
            "avg_candidates": float(metric_dict.get(avg_key, 0.0)),
        }
        if route_name == "hstu":
            row["num_queries"] = float(metric_dict.get("num_hstu_queries", 0.0))
        else:
            row["num_queries"] = ""
        for k in topks:
            row[f"hr@{k}"] = _metric_value(metric_dict, prefix, "hr", k)
            row[f"ndcg@{k}"] = _metric_value(metric_dict, prefix, "ndcg", k)
        rows.append(row)
    return rows


def _rows_to_markdown(split_name: str, split_block: Dict[str, object], topks: Sequence[int]) -> str:
    metric_dict = dict(split_block.get("metrics", {}))
    rows = _route_rows(metric_dict, topks)

    lines: List[str] = []
    lines.append(f"## {split_name.capitalize()}")
    lines.append(
        f"rows={int(split_block.get('num_rows', 0))}, "
        f"eval_points={int(split_block.get('num_eval_points', 0))}, "
        f"unique_users={int(metric_dict.get('num_unique_users', 0.0))}, "
        f"candidate_items={int(metric_dict.get('num_candidates', 0.0))}"
    )
    headers = ["route", "avg_candidates", "num_queries"]
    headers.extend([f"hr@{k}" for k in topks])
    headers.extend([f"ndcg@{k}" for k in topks])
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [
            str(row["route"]),
            _format_float(float(row["avg_candidates"])),
            ("" if row["num_queries"] == "" else str(int(float(row["num_queries"])))),
        ]
        cells.extend(_format_float(float(row[f"hr@{k}"])) for k in topks)
        cells.extend(_format_float(float(row[f"ndcg@{k}"])) for k in topks)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, data: Dict[str, object], topks: Sequence[int]) -> None:
    headers = ["split", "route", "avg_candidates", "num_queries"]
    headers.extend([f"hr@{k}" for k in topks])
    headers.extend([f"ndcg@{k}" for k in topks])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for split_name in ("valid", "test"):
            split_block = data.get(split_name)
            if not isinstance(split_block, dict):
                continue
            metric_dict = dict(split_block.get("metrics", {}))
            for row in _route_rows(metric_dict, topks):
                out = {"split": split_name}
                out.update(row)
                writer.writerow(out)


def main() -> None:
    args = parse_args()
    if args.output_md is None:
        args.output_md = args.input_json.with_name(f"{args.input_json.stem}_table.md")
    if args.output_csv is None:
        args.output_csv = args.input_json.with_name(f"{args.input_json.stem}_table.csv")

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    valid_block = data.get("valid", {})
    metric_dict = dict(valid_block.get("metrics", {}))
    topks = _infer_topks(metric_dict)
    if not topks:
        raise RuntimeError("No hr@K metrics found in input json.")

    md_parts = ["# Recall Route Comparison", ""]
    for split_name in ("valid", "test"):
        split_block = data.get(split_name)
        if not isinstance(split_block, dict):
            continue
        md_parts.append(_rows_to_markdown(split_name, split_block, topks))
    md_text = "\n".join(md_parts).rstrip() + "\n"

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md_text, encoding="utf-8")
    _write_csv(args.output_csv, data, topks)

    print(f"[RouteCompare] markdown saved to {args.output_md}")
    print(f"[RouteCompare] csv saved to {args.output_csv}")
    print(md_text)


if __name__ == "__main__":
    main()
