#!/usr/bin/env python3
"""Build KuaiRand sequence data for HSTU recall."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from dataset import resolve_raw_dataset_spec


LOG_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
]

LOG_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "date": "int32",
    "time_ms": "int64",
    "is_click": "int8",
    "is_like": "int8",
    "is_follow": "int8",
    "is_comment": "int8",
    "is_forward": "int8",
    "is_hate": "int8",
    "long_view": "int8",
}

OUTPUT_COLUMNS = [
    "index",
    "user_id",
    "sequence_item_ids",
    "sequence_ratings",
    "sequence_timestamps",
]


@dataclass(frozen=True)
class DateSplit:
    name: str
    start_date: int
    end_date: int

    def contains(self, date: int) -> bool:
        return self.start_date <= date <= self.end_date


SPLITS = [
    DateSplit("warmup", 20220408, 20220421),
    DateSplit("train", 20220422, 20220501),
    DateSplit("valid", 20220502, 20220505),
    DateSplit("test", 20220506, 20220508),
]


def _split_from_date(date: int) -> Optional[str]:
    for split in SPLITS:
        if split.contains(int(date)):
            return split.name
    return None


def _signal_id(frame: pd.DataFrame) -> np.ndarray:
    is_like = pd.to_numeric(frame["is_like"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_follow = pd.to_numeric(frame["is_follow"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_comment = pd.to_numeric(frame["is_comment"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_forward = pd.to_numeric(frame["is_forward"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    long_view = pd.to_numeric(frame["long_view"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_click = pd.to_numeric(frame["is_click"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_hate = pd.to_numeric(frame["is_hate"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)

    out = np.zeros((len(frame),), dtype=np.int64)
    strong_social = (is_follow == 1) | (is_comment == 1) | (is_forward == 1)
    out[strong_social] = 4
    out[(out == 0) & (is_like == 1)] = 3
    out[(out == 0) & (long_view == 1)] = 2
    out[(out == 0) & (is_click == 1) & (is_hate == 0)] = 1
    return out


def _format_seq(values: Iterable[int]) -> str:
    return ",".join(str(int(v)) for v in values)


def _load_positive_events(
    csv_path: Path,
    nrows: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=LOG_COLUMNS,
        dtype=LOG_DTYPES,
        nrows=nrows,
    )
    df = df.sort_values(["time_ms", "user_id", "video_id"], kind="mergesort").reset_index(drop=True)
    df["split"] = df["date"].map(_split_from_date)
    df = df[df["split"].notnull()].reset_index(drop=True)
    signal_ids = _signal_id(df)
    positive_mask = signal_ids > 0
    if not np.any(positive_mask):
        return df.iloc[0:0][["user_id", "video_id", "time_ms", "date", "split"]].assign(
            signal_id=np.array([], dtype=np.int64)
        )
    df = df.loc[positive_mask, ["user_id", "video_id", "time_ms", "date", "split"]].reset_index(drop=True)
    df["signal_id"] = signal_ids[positive_mask]
    return df[["user_id", "video_id", "time_ms", "date", "split", "signal_id"]].copy()


def _write_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KuaiRand sequence CSVs for HSTU recall.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/KuaiRand-Pure/data"),
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["auto", "1k", "pure"],
        default="pure",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/hstu_interleaved_firsttoken_len100"),
    )
    parser.add_argument("--nrows-warmup", type=int, default=None)
    parser.add_argument("--nrows-main", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = resolve_raw_dataset_spec(args.data_dir, args.dataset_version)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[HSTU Data] loading positive events ...")
    warmup_df = _load_positive_events(args.data_dir / spec.warmup_log, args.nrows_warmup)
    main_df = _load_positive_events(args.data_dir / spec.main_log, args.nrows_main)
    events = pd.concat([warmup_df, main_df], axis=0, ignore_index=True)
    events = events.sort_values(["user_id", "time_ms", "video_id"], kind="mergesort").reset_index(drop=True)
    if len(events) == 0:
        raise RuntimeError("No positive events were found for HSTU data build.")

    user_to_events: Dict[int, List[tuple[int, int, int, str]]] = defaultdict(list)
    for row in events.itertuples(index=False):
        raw_video_id = int(row.video_id)
        shifted_item_id = raw_video_id + 1  # 0 is reserved as padding for HSTU.
        user_to_events[int(row.user_id)].append(
            (shifted_item_id, int(row.signal_id), int(row.time_ms), str(row.split))
        )

    train_rows: List[dict] = []
    valid_rows: List[dict] = []
    test_rows: List[dict] = []
    train_item_ids: set[int] = set()
    valid_item_ids: set[int] = set()
    test_item_ids: set[int] = set()

    for user_id, user_events in user_to_events.items():
        train_items = [item_id for item_id, _, _, split in user_events if split in {"warmup", "train"}]
        train_signals = [signal_id for _, signal_id, _, split in user_events if split in {"warmup", "train"}]
        train_times = [time_ms for _, _, time_ms, split in user_events if split in {"warmup", "train"}]
        if len(train_items) >= 2:
            train_rows.append(
                {
                    "index": len(train_rows),
                    "user_id": int(user_id),
                    "sequence_item_ids": _format_seq(train_items),
                    "sequence_ratings": _format_seq(train_signals),
                    "sequence_timestamps": _format_seq(train_times),
                }
            )
            train_item_ids.update(train_items)

        history_items: List[int] = []
        history_signals: List[int] = []
        history_times: List[int] = []
        for item_id, signal_id, time_ms, split in user_events:
            if split in {"warmup", "train"}:
                history_items.append(item_id)
                history_signals.append(signal_id)
                history_times.append(time_ms)
                continue

            if len(history_items) >= 1:
                row = {
                    "index": 0,
                    "user_id": int(user_id),
                    "sequence_item_ids": _format_seq(history_items + [item_id]),
                    "sequence_ratings": _format_seq(history_signals + [signal_id]),
                    "sequence_timestamps": _format_seq(history_times + [time_ms]),
                }
                if split == "valid":
                    row["index"] = len(valid_rows)
                    valid_rows.append(row)
                    valid_item_ids.add(item_id)
                elif split == "test":
                    row["index"] = len(test_rows)
                    test_rows.append(row)
                    test_item_ids.add(item_id)

            history_items.append(item_id)
            history_signals.append(signal_id)
            history_times.append(time_ms)

    train_path = output_dir / "train.csv"
    valid_path = output_dir / "valid.csv"
    test_path = output_dir / "test.csv"
    _write_rows(train_path, train_rows)
    _write_rows(valid_path, valid_rows)
    _write_rows(test_path, test_rows)

    item_feature_path = args.data_dir / spec.item_features
    all_video_ids = pd.read_csv(item_feature_path, usecols=["video_id"])["video_id"].to_numpy(dtype=np.int64)
    all_video_ids = np.unique(all_video_ids)
    all_hstu_item_ids = all_video_ids + 1

    video_to_hstu = {str(int(video_id)): int(video_id) + 1 for video_id in all_video_ids.tolist()}
    hstu_to_video = {str(int(video_id) + 1): int(video_id) for video_id in all_video_ids.tolist()}
    with (output_dir / "video_id_to_hstu_item_id.json").open("w", encoding="utf-8") as f:
        json.dump(video_to_hstu, f, ensure_ascii=True, indent=2)
    with (output_dir / "hstu_item_id_to_video_id.json").open("w", encoding="utf-8") as f:
        json.dump(hstu_to_video, f, ensure_ascii=True, indent=2)
    np.save(output_dir / "candidate_hstu_item_ids.npy", all_hstu_item_ids.astype(np.int64))
    np.save(output_dir / "train_hstu_item_ids.npy", np.asarray(sorted(train_item_ids), dtype=np.int64))

    max_user_id = max(int(v) for v in user_to_events.keys())
    metadata = {
        "dataset_version": spec.name,
        "data_dir": str(args.data_dir),
        "item_id_offset": 1,
        "signal_mapping": {
            "4": "is_follow OR is_comment OR is_forward",
            "3": "is_like",
            "2": "long_view",
            "1": "is_click AND is_hate == 0",
            "0": "padding",
        },
        "files": {
            "train_csv": str(train_path),
            "valid_csv": str(valid_path),
            "test_csv": str(test_path),
            "candidate_hstu_item_ids": str(output_dir / "candidate_hstu_item_ids.npy"),
            "train_hstu_item_ids": str(output_dir / "train_hstu_item_ids.npy"),
            "video_id_to_hstu_item_id": str(output_dir / "video_id_to_hstu_item_id.json"),
            "hstu_item_id_to_video_id": str(output_dir / "hstu_item_id_to_video_id.json"),
        },
        "stats": {
            "num_users_with_positive_events": int(len(user_to_events)),
            "max_user_id": int(max_user_id),
            "num_items": int(all_hstu_item_ids.max()),
            "num_candidate_items": int(len(all_hstu_item_ids)),
            "train_rows": int(len(train_rows)),
            "valid_rows": int(len(valid_rows)),
            "test_rows": int(len(test_rows)),
            "num_train_items": int(len(train_item_ids)),
            "num_valid_target_items": int(len(valid_item_ids)),
            "num_test_target_items": int(len(test_item_ids)),
        },
        "sequence_protocol": {
            "train": "one full warmup+train positive sequence per user",
            "valid": "one row per valid positive event, using all prior positives as history",
            "test": "one row per test positive event, using all prior positives as history",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    print(json.dumps(metadata["stats"], indent=2))


if __name__ == "__main__":
    main()
