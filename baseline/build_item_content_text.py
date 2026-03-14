#!/usr/bin/env python3
"""Build item content text jsonl for content-based recall."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    return text[:max_chars]


def _load_caption_map(
    csv_path: Path,
    candidate_ids: set[int],
    chunksize: int,
    caption_max_chars: int,
) -> Dict[int, str]:
    out: Dict[int, str] = {}
    usecols = ["final_video_id", "caption"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["final_video_id"] = pd.to_numeric(chunk["final_video_id"], errors="coerce").fillna(0).astype("int64")
        chunk = chunk[chunk["final_video_id"].isin(candidate_ids)]
        if len(chunk) == 0:
            continue
        for row in chunk.itertuples(index=False):
            out[int(row.final_video_id)] = _truncate_text(_safe_text(row.caption), caption_max_chars)
    return out


def _load_category_map(
    csv_path: Path,
    candidate_ids: set[int],
    chunksize: int,
) -> Dict[int, tuple[str, str]]:
    out: Dict[int, tuple[str, str]] = {}
    usecols = [
        "final_video_id",
        "first_level_category_name",
        "second_level_category_name",
    ]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["final_video_id"] = pd.to_numeric(chunk["final_video_id"], errors="coerce").fillna(0).astype("int64")
        chunk = chunk[chunk["final_video_id"].isin(candidate_ids)]
        if len(chunk) == 0:
            continue
        for row in chunk.itertuples(index=False):
            out[int(row.final_video_id)] = (
                _safe_text(row.first_level_category_name),
                _safe_text(row.second_level_category_name),
            )
    return out


def _build_item_text(
    caption: str,
    category_l1: str,
    category_l2: str,
    tag: str,
    video_type: str,
    author_id: str,
) -> str:
    return (
        f"passage: 标题描述: {caption}\n"
        f"一级类目: {category_l1}\n"
        f"二级类目: {category_l2}\n"
        f"标签: {tag}\n"
        f"视频类型: {video_type}\n"
        f"作者: {author_id}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build content text for item embedding.")
    parser.add_argument(
        "--item-features",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/item_features.selected.csv"),
    )
    parser.add_argument(
        "--caption-csv",
        type=Path,
        default=Path("/home/hfx/KuaiRand/KuaiRand-1K/data/kuairand_video_captions.csv"),
    )
    parser.add_argument(
        "--category-csv",
        type=Path,
        default=Path("/home/hfx/KuaiRand/KuaiRand-1K/data/kuairand_video_categories.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/content_assets/item_text.jsonl"),
    )
    parser.add_argument(
        "--output-category-csv",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/content_assets/item_category_features.csv"),
    )
    parser.add_argument("--caption-max-chars", type=int, default=256)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_category_csv.parent.mkdir(parents=True, exist_ok=True)

    video_ids = pd.read_csv(args.item_features, usecols=["video_id"])["video_id"]
    candidate_ids = set(pd.to_numeric(video_ids, errors="coerce").fillna(0).astype("int64").tolist())
    if not candidate_ids:
        raise RuntimeError("No video_id found in item-features.")

    print(f"[content_text] loading captions for {len(candidate_ids)} items ...")
    caption_map = _load_caption_map(
        csv_path=args.caption_csv,
        candidate_ids=candidate_ids,
        chunksize=args.chunksize,
        caption_max_chars=args.caption_max_chars,
    )
    print(f"[content_text] matched captions={len(caption_map)}")

    print(f"[content_text] loading categories for {len(candidate_ids)} items ...")
    category_map = _load_category_map(
        csv_path=args.category_csv,
        candidate_ids=candidate_ids,
        chunksize=args.chunksize,
    )
    print(f"[content_text] matched categories={len(category_map)}")

    usecols = ["video_id", "author_id", "video_type", "tag"]
    rows = 0
    seen_video_ids: set[int] = set()
    with (
        args.output_jsonl.open("w", encoding="utf-8") as f,
        args.output_category_csv.open("w", newline="", encoding="utf-8") as category_f,
    ):
        category_writer = csv.DictWriter(
            category_f,
            fieldnames=["video_id", "category_l1", "category_l2"],
        )
        category_writer.writeheader()
        for chunk in pd.read_csv(args.item_features, usecols=usecols, chunksize=args.chunksize):
            chunk = chunk.drop_duplicates(subset=["video_id"], keep="first")
            for row in chunk.itertuples(index=False):
                video_id = int(row.video_id)
                if video_id in seen_video_ids:
                    continue
                seen_video_ids.add(video_id)
                caption = caption_map.get(video_id, "")
                category_l1, category_l2 = category_map.get(video_id, ("", ""))
                content_text = _build_item_text(
                    caption=caption,
                    category_l1=category_l1,
                    category_l2=category_l2,
                    tag=_safe_text(row.tag),
                    video_type=_safe_text(row.video_type),
                    author_id=_safe_text(row.author_id),
                )
                record = {
                    "video_id": video_id,
                    "content_text": content_text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                category_writer.writerow(
                    {
                        "video_id": video_id,
                        "category_l1": category_l1,
                        "category_l2": category_l2,
                    }
                )
                rows += 1

    print(f"[content_text] saved rows={rows} to {args.output_jsonl}")
    print(f"[content_text] saved category rows={rows} to {args.output_category_csv}")


if __name__ == "__main__":
    main()
