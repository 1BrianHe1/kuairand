#!/usr/bin/env python3
"""Train a lightweight content two-tower recall model for KuaiRand."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data_utils import (
    DEFAULT_BUCKET_SIZES,
    EvalUserSample,
    ItemFeatureStore,
    RecallHardNegativeSampler,
    batched_topk_inner_product_search,
    build_pointwise_eval_samples,
    hit_rate_at_k,
    load_interactions,
    ndcg_at_k,
    parse_history_field,
    parse_topk,
    positive_mask,
    save_json,
    set_seed,
    write_ids_to_npy,
)
from models import ContentTwoTowerRecallModel


LIGHT_CONTEXT_BUCKET_SIZES: Dict[str, int] = {
    "tab": DEFAULT_BUCKET_SIZES["tab"],
    "hour_bucket": DEFAULT_BUCKET_SIZES["hour_bucket"],
    "weekday": 8,
    "is_weekend": 4,
    "short_history_len": 32,
    "short_strong_count": 32,
    "long_strong_history_len": 128,
}

BASE_DATE_WEEKDAY = 4  # 2022-04-08 is Friday, Monday=0.


@dataclass
class ContentEmbeddingStore:
    item_ids: np.ndarray
    item_vecs: np.ndarray

    def lookup_indices(self, video_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        video_ids_arr = np.asarray(list(video_ids), dtype=np.int64)
        if len(self.item_ids) == 0 or len(video_ids_arr) == 0:
            return (
                np.zeros((len(video_ids_arr),), dtype=np.int64),
                np.zeros((len(video_ids_arr),), dtype=bool),
            )
        idx = np.searchsorted(self.item_ids, video_ids_arr)
        idx_clip = np.clip(idx, 0, len(self.item_ids) - 1)
        valid = (idx < len(self.item_ids)) & (self.item_ids[idx_clip] == video_ids_arr)
        out = np.zeros((len(video_ids_arr),), dtype=np.int64)
        if np.any(valid):
            out[valid] = idx[valid]
        return out, valid


@dataclass
class CategoryFeatureStore:
    item_ids: np.ndarray
    category_l1_index: np.ndarray
    category_l2_index: np.ndarray
    num_category_l1: int
    num_category_l2: int

    def lookup_category_indices(self, video_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        video_ids_arr = np.asarray(list(video_ids), dtype=np.int64)
        if len(self.item_ids) == 0 or len(video_ids_arr) == 0:
            return (
                np.zeros((len(video_ids_arr),), dtype=np.int64),
                np.zeros((len(video_ids_arr),), dtype=np.int64),
                np.zeros((len(video_ids_arr),), dtype=bool),
            )
        idx = np.searchsorted(self.item_ids, video_ids_arr)
        idx_clip = np.clip(idx, 0, len(self.item_ids) - 1)
        valid = (idx < len(self.item_ids)) & (self.item_ids[idx_clip] == video_ids_arr)
        out_l1 = np.zeros((len(video_ids_arr),), dtype=np.int64)
        out_l2 = np.zeros((len(video_ids_arr),), dtype=np.int64)
        if np.any(valid):
            out_l1[valid] = self.category_l1_index[idx[valid]]
            out_l2[valid] = self.category_l2_index[idx[valid]]
        return out_l1, out_l2, valid


@dataclass
class ContentRecallTrainQuerySample:
    tab: int
    hour_bucket: int
    date_bucket: int
    time_ms: int
    content_history_video_ids: str
    content_history_signal_types: str
    content_history_time_ms: str
    strong_history_video_ids: str
    strong_history_signal_types: str
    strong_history_time_ms: str
    negative_history_video_ids: str
    negative_history_signal_types: str
    negative_history_time_ms: str
    positive_video_ids: np.ndarray
    explicit_negative_video_ids: np.ndarray


def _serialize_negative_history(
    history: list[tuple[int, int, int]] | Sequence[tuple[int, int, int]],
    max_history_len: int,
) -> tuple[str, str, str, int]:
    if max_history_len <= 0 or len(history) == 0:
        return "", "", "", 0
    selected: List[tuple[int, int, int]] = []
    seen_video_ids: set[int] = set()
    for video_id, event_time_ms, signal_type in reversed(history):
        if int(video_id) in seen_video_ids:
            continue
        selected.append((int(video_id), int(event_time_ms), int(signal_type)))
        seen_video_ids.add(int(video_id))
        if len(selected) >= max_history_len:
            break
    if not selected:
        return "", "", "", 0
    video_text = ",".join(str(video_id) for video_id, _, _ in selected)
    signal_text = ",".join(str(signal_type) for _, _, signal_type in selected)
    time_text = ",".join(str(event_time_ms) for _, event_time_ms, _ in selected)
    return video_text, signal_text, time_text, len(selected)


class ContentRecallTrainQueryDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        positive_label_mode: str,
        negative_history_len: int,
        sample_frac: float = 1.0,
        seed: int = 42,
    ) -> None:
        query_frame = frame.copy()
        query_frame = query_frame.sort_values(["user_id", "time_ms"], kind="mergesort")
        pos = positive_mask(query_frame, positive_label_mode)
        query_frame["is_positive"] = pos.astype(np.int8)

        samples: List[ContentRecallTrainQuerySample] = []
        negative_history_len = max(int(negative_history_len), 0)
        negative_histories: Dict[int, List[tuple[int, int, int]]] = {}
        for (_, _), group in query_frame.groupby(["user_id", "time_ms"], sort=False):
            positives = np.unique(
                group.loc[group["is_positive"] == 1, "video_id"].to_numpy(dtype=np.int64)
            )
            if len(positives) == 0:
                if negative_history_len > 0:
                    user_id = int(group.iloc[0]["user_id"])
                    history = negative_histories.setdefault(user_id, [])
                    group_time_ms = int(group.iloc[0]["time_ms"])
                    strong_negative_ids = np.unique(
                        group.loc[group["is_hate"].fillna(0).astype(np.int8) == 1, "video_id"].to_numpy(dtype=np.int64)
                    ).tolist()
                    for video_id in strong_negative_ids:
                        history.append((int(video_id), group_time_ms, 2))
                    if len(history) > negative_history_len:
                        negative_histories[user_id] = history[-negative_history_len:]
                continue
            negatives = np.unique(
                group.loc[group["is_positive"] == 0, "video_id"].to_numpy(dtype=np.int64)
            )
            first = group.iloc[0]
            user_id = int(first["user_id"])
            negative_video_ids, negative_signal_types, negative_time_ms, _ = _serialize_negative_history(
                negative_histories.get(user_id, []),
                max_history_len=negative_history_len,
            )
            samples.append(
                ContentRecallTrainQuerySample(
                    tab=int(first["tab"]),
                    hour_bucket=int(first["hour_bucket"]),
                    date_bucket=int(first["date_bucket"]),
                    time_ms=int(first.get("time_ms", 0)),
                    content_history_video_ids=str(first.get("content_history_video_ids", "")),
                    content_history_signal_types=str(first.get("content_history_signal_types", "")),
                    content_history_time_ms=str(first.get("content_history_time_ms", "")),
                    strong_history_video_ids=str(first.get("strong_history_video_ids", "")),
                    strong_history_signal_types=str(first.get("strong_history_signal_types", "")),
                    strong_history_time_ms=str(first.get("strong_history_time_ms", "")),
                    negative_history_video_ids=negative_video_ids,
                    negative_history_signal_types=negative_signal_types,
                    negative_history_time_ms=negative_time_ms,
                    positive_video_ids=positives,
                    explicit_negative_video_ids=negatives,
                )
            )
            if negative_history_len > 0:
                history = negative_histories.setdefault(user_id, [])
                group_time_ms = int(first["time_ms"])
                weak_negative_ids = np.unique(
                    group.loc[
                        (group["is_positive"] == 0) & (group["is_hate"].fillna(0).astype(np.int8) == 0),
                        "video_id",
                    ].to_numpy(dtype=np.int64)
                ).tolist()
                strong_negative_ids = np.unique(
                    group.loc[group["is_hate"].fillna(0).astype(np.int8) == 1, "video_id"].to_numpy(dtype=np.int64)
                ).tolist()
                for video_id in weak_negative_ids:
                    history.append((int(video_id), group_time_ms, 1))
                for video_id in strong_negative_ids:
                    history.append((int(video_id), group_time_ms, 2))
                if len(history) > negative_history_len:
                    negative_histories[user_id] = history[-negative_history_len:]

        if sample_frac < 1.0 and samples:
            rng = np.random.default_rng(seed)
            keep = rng.random(len(samples)) < sample_frac
            samples = [sample for sample, use in zip(samples, keep.tolist()) if use]

        self.samples = samples
        self.avg_positive_items = (
            float(np.mean([len(sample.positive_video_ids) for sample in samples])) if samples else 0.0
        )
        self.avg_explicit_negative_items = (
            float(np.mean([len(sample.explicit_negative_video_ids) for sample in samples])) if samples else 0.0
        )
        self.avg_negative_history_items = (
            float(
                np.mean(
                    [
                        len(parse_history_field(sample.negative_history_video_ids, max_history_len=0))
                        for sample in samples
                    ]
                )
            )
            if samples
            else 0.0
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "tab": sample.tab,
            "hour_bucket": sample.hour_bucket,
            "date_bucket": sample.date_bucket,
            "time_ms": sample.time_ms,
            "content_history_video_ids": sample.content_history_video_ids,
            "content_history_signal_types": sample.content_history_signal_types,
            "content_history_time_ms": sample.content_history_time_ms,
            "strong_history_video_ids": sample.strong_history_video_ids,
            "strong_history_signal_types": sample.strong_history_signal_types,
            "strong_history_time_ms": sample.strong_history_time_ms,
            "negative_history_video_ids": sample.negative_history_video_ids,
            "negative_history_signal_types": sample.negative_history_signal_types,
            "negative_history_time_ms": sample.negative_history_time_ms,
            "positive_video_ids": sample.positive_video_ids,
            "explicit_negative_video_ids": sample.explicit_negative_video_ids,
        }


def load_content_embedding_store(
    embedding_path: Path,
    video_id_to_index_path: Path,
    candidate_item_ids: np.ndarray | None,
) -> ContentEmbeddingStore:
    with video_id_to_index_path.open("r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    mapping = {int(k): int(v) for k, v in raw_mapping.items()}
    emb = np.load(embedding_path).astype(np.float32, copy=False)

    if candidate_item_ids is None:
        item_ids = np.asarray(sorted(mapping.keys()), dtype=np.int64)
    else:
        candidate_arr = np.unique(np.asarray(candidate_item_ids, dtype=np.int64))
        item_ids = np.asarray([int(v) for v in candidate_arr.tolist() if int(v) in mapping], dtype=np.int64)

    if len(item_ids) == 0:
        return ContentEmbeddingStore(
            item_ids=np.zeros((0,), dtype=np.int64),
            item_vecs=np.zeros((0, emb.shape[1]), dtype=np.float32),
        )

    indices = np.asarray([mapping[int(video_id)] for video_id in item_ids.tolist()], dtype=np.int64)
    vecs = np.ascontiguousarray(emb[indices], dtype=np.float32)
    return ContentEmbeddingStore(item_ids=item_ids, item_vecs=vecs)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def load_category_feature_store(
    category_asset_path: Path,
    raw_category_csv: Path,
    candidate_item_ids: np.ndarray,
    chunksize: int = 250_000,
) -> CategoryFeatureStore:
    if category_asset_path.exists():
        df = pd.read_csv(category_asset_path, usecols=["video_id", "category_l1", "category_l2"])
        df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce").fillna(0).astype("int64")
        df = df[df["video_id"].isin(set(candidate_item_ids.tolist()))]
    else:
        if not raw_category_csv.exists():
            raise FileNotFoundError(
                f"Category asset not found at {category_asset_path} and raw category csv not found at {raw_category_csv}."
            )
        rows: List[pd.DataFrame] = []
        candidate_set = set(candidate_item_ids.tolist())
        usecols = [
            "final_video_id",
            "first_level_category_name",
            "second_level_category_name",
        ]
        for chunk in pd.read_csv(raw_category_csv, usecols=usecols, chunksize=chunksize):
            chunk["final_video_id"] = pd.to_numeric(chunk["final_video_id"], errors="coerce").fillna(0).astype("int64")
            chunk = chunk[chunk["final_video_id"].isin(candidate_set)]
            if len(chunk) == 0:
                continue
            chunk = chunk.rename(
                columns={
                    "final_video_id": "video_id",
                    "first_level_category_name": "category_l1",
                    "second_level_category_name": "category_l2",
                }
            )
            rows.append(chunk)
        df = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
            columns=["video_id", "category_l1", "category_l2"]
        )

    if len(df) == 0:
        return CategoryFeatureStore(
            item_ids=np.zeros((0,), dtype=np.int64),
            category_l1_index=np.zeros((0,), dtype=np.int64),
            category_l2_index=np.zeros((0,), dtype=np.int64),
            num_category_l1=0,
            num_category_l2=0,
        )

    df = df.fillna("")
    df = df.drop_duplicates(subset=["video_id"], keep="first")
    df = df.sort_values("video_id", kind="mergesort").reset_index(drop=True)
    item_ids = df["video_id"].to_numpy(dtype=np.int64)
    l1_values = [_safe_text(v) for v in df["category_l1"].tolist()]
    l2_values = [_safe_text(v) for v in df["category_l2"].tolist()]

    l1_vocab = sorted({v for v in l1_values if v})
    l2_vocab = sorted({v for v in l2_values if v})
    l1_map = {value: idx + 1 for idx, value in enumerate(l1_vocab)}
    l2_map = {value: idx + 1 for idx, value in enumerate(l2_vocab)}

    l1_index = np.asarray([l1_map.get(v, 0) for v in l1_values], dtype=np.int64)
    l2_index = np.asarray([l2_map.get(v, 0) for v in l2_values], dtype=np.int64)
    return CategoryFeatureStore(
        item_ids=item_ids,
        category_l1_index=l1_index,
        category_l2_index=l2_index,
        num_category_l1=len(l1_vocab),
        num_category_l2=len(l2_vocab),
    )


def parse_args() -> argparse.Namespace:
    default_num_workers = max(1, min(8, (os.cpu_count() or 4) // 2))
    parser = argparse.ArgumentParser(description="Train KuaiRand lightweight content two-tower recall model.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure"),
    )
    parser.add_argument(
        "--content-item-emb",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/content_assets/item_content_emb.npy"),
    )
    parser.add_argument(
        "--content-video-id-to-index",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/content_assets/video_id_to_index.json"),
    )
    parser.add_argument(
        "--content-category-csv",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/content_assets/item_category_features.csv"),
    )
    parser.add_argument(
        "--raw-category-csv",
        type=Path,
        default=Path("/home/hfx/KuaiRand/KuaiRand-1K/data/kuairand_video_categories.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/content_recall_pure"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--num-explicit-negatives", type=int, default=4)
    parser.add_argument("--num-hard-negatives", type=int, default=8)
    parser.add_argument("--num-random-negatives", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--tower-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--short-history-len",
        "--max-history-len",
        dest="short_history_len",
        type=int,
        default=15,
    )
    parser.add_argument("--long-term-history-len", type=int, default=100)
    parser.add_argument(
        "--negative-immunity-mode",
        type=str,
        default="off",
        choices=["off", "fixed", "gated"],
    )
    parser.add_argument("--negative-history-len", type=int, default=10)
    parser.add_argument("--negative-half-life-hours", type=float, default=72.0)
    parser.add_argument("--negative-fixed-alpha", type=float, default=0.5)
    parser.add_argument("--negative-max-alpha", type=float, default=1.0)
    parser.add_argument("--negative-gate-hidden-dim", type=int, default=64)
    parser.add_argument("--negative-relative-weight", type=float, default=0.35)
    parser.add_argument("--negative-hate-weight", type=float, default=1.0)
    parser.add_argument("--negative-semantic-sim-threshold", type=float, default=0.1)
    parser.add_argument("--max-positive-items-per-query", type=int, default=4)
    parser.add_argument(
        "--label-mode",
        choices=["click", "long_view", "click_or_long", "signal_positive"],
        default="click_or_long",
    )
    parser.add_argument("--train-max-rows", type=int, default=None)
    parser.add_argument("--train-sample-frac", type=float, default=1.0)
    parser.add_argument("--valid-max-rows", type=int, default=None)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--recall-topn", type=int, default=300)
    parser.add_argument("--eval-item-batch-size", type=int, default=4096)
    parser.add_argument("--eval-query-batch-size", type=int, default=64)
    parser.add_argument("--click-weight", type=float, default=0.5)
    parser.add_argument("--long-view-weight", type=float, default=1.0)
    parser.add_argument("--like-weight", type=float, default=1.25)
    parser.add_argument("--social-weight", type=float, default=1.5)
    parser.add_argument("--short-history-half-life-hours", type=float, default=48.0)
    parser.add_argument("--topk", type=str, default="50,100,200")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--skip-no-candidate-positive", action="store_true")
    return parser.parse_args()


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _multi_positive_sampled_softmax_loss(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, int, float]:
    valid_mask = candidate_mask > 0.5
    positive_mask = positive_mask > 0.5
    active_mask = positive_mask.any(dim=1) & (valid_mask.sum(dim=1) > positive_mask.sum(dim=1))
    if active_mask.sum().item() == 0:
        return logits.new_zeros(()), 0, 0.0

    masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    pos_logits = masked_logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    log_num = torch.logsumexp(pos_logits[active_mask], dim=1)
    log_denom = torch.logsumexp(masked_logits[active_mask], dim=1)
    loss = -(log_num - log_denom).mean()
    avg_candidates = float(valid_mask[active_mask].sum(dim=1).float().mean().item())
    return loss, int(active_mask.sum().item()), avg_candidates


def _signal_weight(signal_code: int, weight_map: Dict[int, float]) -> float:
    return float(weight_map.get(int(signal_code), 0.0))


def _weekday_from_date_bucket(date_bucket: np.ndarray) -> np.ndarray:
    out = (BASE_DATE_WEEKDAY + date_bucket.astype(np.int64)) % 7
    out = np.clip(out, a_min=0, a_max=6)
    return out.astype(np.int64)


def _encode_light_context_features(
    tab: np.ndarray,
    hour_bucket: np.ndarray,
    date_bucket: np.ndarray,
    short_history_len: np.ndarray,
    short_strong_count: np.ndarray,
    long_strong_history_len: np.ndarray,
) -> np.ndarray:
    weekday = _weekday_from_date_bucket(date_bucket)
    is_weekend = ((weekday == 5) | (weekday == 6)).astype(np.int64)
    out = np.zeros((len(tab), len(LIGHT_CONTEXT_BUCKET_SIZES)), dtype=np.int64)
    out[:, 0] = np.clip(tab.astype(np.int64), 0, LIGHT_CONTEXT_BUCKET_SIZES["tab"] - 1)
    out[:, 1] = np.clip(hour_bucket.astype(np.int64), 0, LIGHT_CONTEXT_BUCKET_SIZES["hour_bucket"] - 1)
    out[:, 2] = np.clip(weekday, 0, LIGHT_CONTEXT_BUCKET_SIZES["weekday"] - 1)
    out[:, 3] = np.clip(is_weekend, 0, LIGHT_CONTEXT_BUCKET_SIZES["is_weekend"] - 1)
    out[:, 4] = np.clip(short_history_len.astype(np.int64), 0, LIGHT_CONTEXT_BUCKET_SIZES["short_history_len"] - 1)
    out[:, 5] = np.clip(short_strong_count.astype(np.int64), 0, LIGHT_CONTEXT_BUCKET_SIZES["short_strong_count"] - 1)
    out[:, 6] = np.clip(
        long_strong_history_len.astype(np.int64),
        0,
        LIGHT_CONTEXT_BUCKET_SIZES["long_strong_history_len"] - 1,
    )
    return out


def _build_short_term_features(
    *,
    raw_item_ids: str,
    raw_signal_types: str,
    raw_event_times: str,
    query_time_ms: int,
    content_store: ContentEmbeddingStore,
    short_history_len: int,
    signal_weight_map: Dict[int, float],
    half_life_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    content_dim = int(content_store.item_vecs.shape[1]) if content_store.item_vecs.ndim == 2 else 0
    hist_emb = np.zeros((short_history_len, content_dim), dtype=np.float32)
    hist_weights = np.zeros((short_history_len,), dtype=np.float32)
    hist_mask = np.zeros((short_history_len,), dtype=np.float32)

    item_ids = parse_history_field(raw_item_ids, max_history_len=0)
    signal_types = parse_history_field(raw_signal_types, max_history_len=0)
    event_times = parse_history_field(raw_event_times, max_history_len=0)
    take = min(len(item_ids), len(signal_types), len(event_times), short_history_len)
    if take <= 0 or content_dim == 0:
        return hist_emb, hist_weights, hist_mask, np.zeros((0,), dtype=np.int64), 0, 0

    item_ids = item_ids[:take]
    signal_types = signal_types[:take]
    event_times = event_times[:take]
    item_idx, valid = content_store.lookup_indices(item_ids)
    if not np.any(valid):
        return hist_emb, hist_weights, hist_mask, np.zeros((0,), dtype=np.int64), 0, 0

    kept_ids = np.asarray([item_ids[i] for i, keep in enumerate(valid.tolist()) if keep], dtype=np.int64)
    kept_idx = item_idx[valid]
    kept_signal = [signal_types[i] for i, keep in enumerate(valid.tolist()) if keep]
    kept_times = [event_times[i] for i, keep in enumerate(valid.tolist()) if keep]
    kept = min(len(kept_idx), short_history_len)
    hist_emb[:kept] = content_store.item_vecs[kept_idx[:kept]]
    hist_mask[:kept] = 1.0
    for i in range(kept):
        base_weight = _signal_weight(kept_signal[i], signal_weight_map)
        if half_life_ms > 0.0:
            age_ms = max(0.0, float(query_time_ms) - float(kept_times[i]))
            decay = 0.5 ** (age_ms / half_life_ms)
        else:
            decay = 1.0
        hist_weights[i] = float(base_weight * decay)
    strong_count = int(sum(1 for value in kept_signal[:kept] if int(value) >= 2))
    return hist_emb, hist_weights, hist_mask, kept_ids[:kept], kept, strong_count


def _build_long_term_preference(
    *,
    raw_item_ids: str,
    raw_signal_types: str,
    category_store: CategoryFeatureStore,
    long_term_history_len: int,
    signal_weight_map: Dict[int, float],
) -> tuple[np.ndarray, int]:
    dim = int(category_store.num_category_l1 + category_store.num_category_l2)
    if dim == 0 or long_term_history_len <= 0:
        return np.zeros((dim,), dtype=np.float32), 0

    item_ids = parse_history_field(raw_item_ids, max_history_len=0)
    signal_types = parse_history_field(raw_signal_types, max_history_len=0)
    take = min(len(item_ids), len(signal_types), long_term_history_len)
    if take <= 0:
        return np.zeros((dim,), dtype=np.float32), 0

    item_ids = item_ids[:take]
    signal_types = signal_types[:take]
    l1_idx, l2_idx, valid = category_store.lookup_category_indices(item_ids)
    if not np.any(valid):
        return np.zeros((dim,), dtype=np.float32), 0

    pref = np.zeros((dim,), dtype=np.float32)
    valid_count = 0
    for i, keep in enumerate(valid.tolist()):
        if not keep:
            continue
        weight = _signal_weight(signal_types[i], signal_weight_map)
        if weight <= 0.0:
            continue
        if int(l1_idx[i]) > 0:
            pref[int(l1_idx[i]) - 1] += float(weight)
        if int(l2_idx[i]) > 0:
            pref[category_store.num_category_l1 + int(l2_idx[i]) - 1] += float(weight)
        valid_count += 1

    l1_sum = float(pref[: category_store.num_category_l1].sum())
    l2_sum = float(pref[category_store.num_category_l1 :].sum())
    if l1_sum > 0:
        pref[: category_store.num_category_l1] /= l1_sum
    if l2_sum > 0:
        pref[category_store.num_category_l1 :] /= l2_sum
    return pref, valid_count


def _build_negative_history_features(
    *,
    raw_item_ids: str,
    raw_signal_types: str,
    raw_event_times: str,
    query_time_ms: int,
    content_store: ContentEmbeddingStore,
    negative_history_len: int,
    half_life_ms: float,
    signal_weight_map: Dict[int, float],
    semantic_similarity_threshold: float,
    reference_hist_emb: np.ndarray | None = None,
    reference_hist_weights: np.ndarray | None = None,
    reference_hist_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    content_dim = int(content_store.item_vecs.shape[1]) if content_store.item_vecs.ndim == 2 else 0
    hist_emb = np.zeros((negative_history_len, content_dim), dtype=np.float32)
    hist_weights = np.zeros((negative_history_len,), dtype=np.float32)
    hist_mask = np.zeros((negative_history_len,), dtype=np.float32)
    if negative_history_len <= 0 or content_dim == 0:
        return hist_emb, hist_weights, hist_mask, 0, 0.0

    item_ids = parse_history_field(raw_item_ids, max_history_len=0)
    signal_types = parse_history_field(raw_signal_types, max_history_len=0)
    event_times = parse_history_field(raw_event_times, max_history_len=0)
    take = min(len(item_ids), len(signal_types), len(event_times), negative_history_len)
    if take <= 0:
        return hist_emb, hist_weights, hist_mask, 0, 0.0

    item_ids = item_ids[:take]
    signal_types = signal_types[:take]
    event_times = event_times[:take]
    item_idx, valid = content_store.lookup_indices(item_ids)
    if not np.any(valid):
        return hist_emb, hist_weights, hist_mask, 0, 0.0

    kept_idx = item_idx[valid]
    kept_signal_types = [signal_types[i] for i, keep in enumerate(valid.tolist()) if keep]
    kept_times = [event_times[i] for i, keep in enumerate(valid.tolist()) if keep]
    kept_emb = np.ascontiguousarray(content_store.item_vecs[kept_idx], dtype=np.float32)
    reference_vec = None
    if (
        reference_hist_emb is not None
        and reference_hist_weights is not None
        and reference_hist_mask is not None
        and reference_hist_emb.size > 0
    ):
        ref_weight = (reference_hist_weights * reference_hist_mask).astype(np.float32, copy=False)
        denom = float(ref_weight.sum())
        if denom > 1e-6:
            reference_vec = (reference_hist_emb * ref_weight[:, None]).sum(axis=0) / denom
            ref_norm = float(np.linalg.norm(reference_vec))
            if ref_norm > 1e-6:
                reference_vec = reference_vec / ref_norm
            else:
                reference_vec = None

    candidates: List[tuple[float, np.ndarray, float]] = []
    recentness = 0.0
    for i in range(len(kept_idx)):
        signal_type = int(kept_signal_types[i])
        base_weight = float(signal_weight_map.get(signal_type, 0.0))
        if base_weight <= 0.0:
            continue
        if half_life_ms > 0.0:
            age_ms = max(0.0, float(query_time_ms) - float(kept_times[i]))
            decay = 0.5 ** (age_ms / half_life_ms)
        else:
            decay = 1.0
        semantic_weight = 1.0
        if signal_type == 1:
            if reference_vec is None:
                continue
            item_vec = kept_emb[i]
            item_norm = float(np.linalg.norm(item_vec))
            if item_norm <= 1e-6:
                continue
            sim = float(np.dot(item_vec / item_norm, reference_vec))
            if sim < float(semantic_similarity_threshold):
                continue
            semantic_weight = sim
        weight = float(base_weight * decay * semantic_weight)
        if weight <= 0.0:
            continue
        candidates.append((weight, kept_emb[i], float(decay)))
        recentness = max(recentness, float(decay))
    if not candidates:
        return hist_emb, hist_weights, hist_mask, 0, 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept = min(len(candidates), negative_history_len)
    for i in range(kept):
        weight, item_emb, _ = candidates[i]
        hist_emb[i] = item_emb
        hist_weights[i] = float(weight)
        hist_mask[i] = 1.0
    return hist_emb, hist_weights, hist_mask, kept, recentness


def compute_content_immunity_metrics(
    pred_lists: Sequence[Sequence[int]],
    samples: Sequence[EvalUserSample],
    category_store: CategoryFeatureStore,
    topk: Sequence[int],
    negative_history_len: int,
    prefix: str = "",
) -> Dict[str, float]:
    if len(samples) == 0:
        out = {
            f"{prefix}recent_hate_rate": 0.0,
            f"{prefix}num_recent_hate_samples": 0.0,
            f"{prefix}num_no_recent_hate_samples": 0.0,
        }
        for k in topk:
            out[f"{prefix}hr_recent_hate@{k}"] = 0.0
            out[f"{prefix}ndcg_recent_hate@{k}"] = 0.0
            out[f"{prefix}hr_no_recent_hate@{k}"] = 0.0
            out[f"{prefix}ndcg_no_recent_hate@{k}"] = 0.0
            out[f"{prefix}hate_l1_leakage@{k}"] = 0.0
            out[f"{prefix}hate_l2_leakage@{k}"] = 0.0
        return out

    recent_hate_mask = np.zeros((len(samples),), dtype=bool)
    hate_l1_sets: List[set[int]] = []
    hate_l2_sets: List[set[int]] = []
    for idx, sample in enumerate(samples):
        if negative_history_len <= 0:
            neg_item_ids = []
        else:
            neg_item_ids = parse_history_field(
                sample.negative_history_video_ids,
                max_history_len=negative_history_len,
            )
        if len(neg_item_ids) == 0:
            hate_l1_sets.append(set())
            hate_l2_sets.append(set())
            continue
        l1_idx, l2_idx, valid = category_store.lookup_category_indices(neg_item_ids)
        l1_set = {int(v) for v, keep in zip(l1_idx.tolist(), valid.tolist()) if keep and int(v) > 0}
        l2_set = {int(v) for v, keep in zip(l2_idx.tolist(), valid.tolist()) if keep and int(v) > 0}
        hate_l1_sets.append(l1_set)
        hate_l2_sets.append(l2_set)
        recent_hate_mask[idx] = bool(l1_set or l2_set)

    recent_count = int(recent_hate_mask.sum())
    no_recent_count = int(len(samples) - recent_count)
    out = {
        f"{prefix}recent_hate_rate": float(recent_count / max(len(samples), 1)),
        f"{prefix}num_recent_hate_samples": float(recent_count),
        f"{prefix}num_no_recent_hate_samples": float(no_recent_count),
    }
    for k in topk:
        hr_recent = 0.0
        ndcg_recent = 0.0
        hr_no_recent = 0.0
        ndcg_no_recent = 0.0
        leakage_l1 = 0.0
        leakage_l2 = 0.0
        for idx, sample in enumerate(samples):
            pred = list(pred_lists[idx][:k])
            if recent_hate_mask[idx]:
                hr_recent += hit_rate_at_k(pred, gt_items=sample.positives, k=k)
                ndcg_recent += ndcg_at_k(pred, gt_items=sample.positives, k=k)
                if len(pred) > 0:
                    pred_l1, pred_l2, valid = category_store.lookup_category_indices(pred)
                    kept_l1 = [int(v) for v, keep in zip(pred_l1.tolist(), valid.tolist()) if keep and int(v) > 0]
                    kept_l2 = [int(v) for v, keep in zip(pred_l2.tolist(), valid.tolist()) if keep and int(v) > 0]
                    if kept_l1:
                        leakage_l1 += float(
                            sum(1 for value in kept_l1 if value in hate_l1_sets[idx]) / len(kept_l1)
                        )
                    if kept_l2:
                        leakage_l2 += float(
                            sum(1 for value in kept_l2 if value in hate_l2_sets[idx]) / len(kept_l2)
                        )
            else:
                hr_no_recent += hit_rate_at_k(pred, gt_items=sample.positives, k=k)
                ndcg_no_recent += ndcg_at_k(pred, gt_items=sample.positives, k=k)
        out[f"{prefix}hr_recent_hate@{k}"] = float(hr_recent / recent_count) if recent_count > 0 else 0.0
        out[f"{prefix}ndcg_recent_hate@{k}"] = float(ndcg_recent / recent_count) if recent_count > 0 else 0.0
        out[f"{prefix}hr_no_recent_hate@{k}"] = float(hr_no_recent / no_recent_count) if no_recent_count > 0 else 0.0
        out[f"{prefix}ndcg_no_recent_hate@{k}"] = (
            float(ndcg_no_recent / no_recent_count) if no_recent_count > 0 else 0.0
        )
        out[f"{prefix}hate_l1_leakage@{k}"] = float(leakage_l1 / recent_count) if recent_count > 0 else 0.0
        out[f"{prefix}hate_l2_leakage@{k}"] = float(leakage_l2 / recent_count) if recent_count > 0 else 0.0
    return out


class ContentRecallQueryBatchCollator:
    def __init__(
        self,
        content_store: ContentEmbeddingStore,
        category_store: CategoryFeatureStore,
        short_history_len: int,
        long_term_history_len: int,
        negative_history_len: int,
        short_signal_weight_map: Dict[int, float],
        long_signal_weight_map: Dict[int, float],
        negative_signal_weight_map: Dict[int, float],
        negative_semantic_similarity_threshold: float,
        short_half_life_hours: float,
        negative_half_life_hours: float,
        hard_negative_sampler: RecallHardNegativeSampler,
        num_explicit_negatives: int,
        num_hard_negatives: int,
        num_random_negatives: int,
        max_positive_items: int,
        seed: int,
    ) -> None:
        self.content_store = content_store
        self.category_store = category_store
        self.short_history_len = int(short_history_len)
        self.long_term_history_len = int(long_term_history_len)
        self.negative_history_len = int(negative_history_len)
        self.short_signal_weight_map = dict(short_signal_weight_map)
        self.long_signal_weight_map = dict(long_signal_weight_map)
        self.negative_signal_weight_map = dict(negative_signal_weight_map)
        self.negative_semantic_similarity_threshold = float(negative_semantic_similarity_threshold)
        self.short_half_life_ms = max(float(short_half_life_hours), 0.0) * 3600.0 * 1000.0
        self.negative_half_life_ms = max(float(negative_half_life_hours), 0.0) * 3600.0 * 1000.0
        self.hard_negative_sampler = hard_negative_sampler
        self.num_explicit_negatives = int(num_explicit_negatives)
        self.num_hard_negatives = int(num_hard_negatives)
        self.num_random_negatives = int(num_random_negatives)
        self.max_positive_items = int(max_positive_items)
        self.seed = int(seed)
        self._call_count = 0
        self.content_dim = int(content_store.item_vecs.shape[1]) if content_store.item_vecs.ndim == 2 else 0
        self.long_term_dim = int(category_store.num_category_l1 + category_store.num_category_l2)

    def _make_rng(self) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        self._call_count += 1
        return np.random.default_rng(self.seed + worker_id * 1_000_003 + self._call_count)

    @staticmethod
    def _sample_without_replacement(
        values: np.ndarray,
        max_take: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        if len(values) <= max_take:
            return values
        idx = rng.choice(len(values), size=max_take, replace=False)
        return values[idx]

    def __call__(self, batch: List[dict]) -> dict:
        rng = self._make_rng()

        tab = np.asarray([x["tab"] for x in batch], dtype=np.int64)
        hour_bucket = np.asarray([x["hour_bucket"] for x in batch], dtype=np.int64)
        date_bucket = np.asarray([x["date_bucket"] for x in batch], dtype=np.int64)
        query_time_ms = np.asarray([x["time_ms"] for x in batch], dtype=np.int64)

        hist_content_emb = np.zeros((len(batch), self.short_history_len, self.content_dim), dtype=np.float32)
        hist_weights = np.zeros((len(batch), self.short_history_len), dtype=np.float32)
        hist_mask = np.zeros((len(batch), self.short_history_len), dtype=np.float32)
        neg_hist_content_emb = np.zeros((len(batch), self.negative_history_len, self.content_dim), dtype=np.float32)
        neg_hist_weights = np.zeros((len(batch), self.negative_history_len), dtype=np.float32)
        neg_hist_mask = np.zeros((len(batch), self.negative_history_len), dtype=np.float32)
        neg_history_stats = np.zeros((len(batch), 2), dtype=np.float32)
        long_term_pref = np.zeros((len(batch), self.long_term_dim), dtype=np.float32)
        short_history_counts = np.zeros((len(batch),), dtype=np.int64)
        short_strong_counts = np.zeros((len(batch),), dtype=np.int64)
        long_strong_counts = np.zeros((len(batch),), dtype=np.int64)
        short_history_video_ids: List[np.ndarray] = []

        for i, sample in enumerate(batch):
            (
                hist_content_emb[i],
                hist_weights[i],
                hist_mask[i],
                kept_ids,
                short_history_counts[i],
                short_strong_counts[i],
            ) = _build_short_term_features(
                raw_item_ids=str(sample.get("content_history_video_ids", "")),
                raw_signal_types=str(sample.get("content_history_signal_types", "")),
                raw_event_times=str(sample.get("content_history_time_ms", "")),
                query_time_ms=int(query_time_ms[i]),
                content_store=self.content_store,
                short_history_len=self.short_history_len,
                signal_weight_map=self.short_signal_weight_map,
                half_life_ms=self.short_half_life_ms,
            )
            (
                neg_hist_content_emb[i],
                neg_hist_weights[i],
                neg_hist_mask[i],
                neg_count,
                neg_recentness,
            ) = _build_negative_history_features(
                raw_item_ids=str(sample.get("negative_history_video_ids", "")),
                raw_signal_types=str(sample.get("negative_history_signal_types", "")),
                raw_event_times=str(sample.get("negative_history_time_ms", "")),
                query_time_ms=int(query_time_ms[i]),
                content_store=self.content_store,
                negative_history_len=self.negative_history_len,
                half_life_ms=self.negative_half_life_ms,
                signal_weight_map=self.negative_signal_weight_map,
                semantic_similarity_threshold=self.negative_semantic_similarity_threshold,
                reference_hist_emb=hist_content_emb[i],
                reference_hist_weights=hist_weights[i],
                reference_hist_mask=hist_mask[i],
            )
            neg_history_stats[i, 0] = (
                float(neg_count) / float(max(self.negative_history_len, 1))
                if self.negative_history_len > 0
                else 0.0
            )
            neg_history_stats[i, 1] = float(neg_recentness)
            long_term_pref[i], long_strong_counts[i] = _build_long_term_preference(
                raw_item_ids=str(sample.get("strong_history_video_ids", "")),
                raw_signal_types=str(sample.get("strong_history_signal_types", "")),
                category_store=self.category_store,
                long_term_history_len=self.long_term_history_len,
                signal_weight_map=self.long_signal_weight_map,
            )
            short_history_video_ids.append(kept_ids)

        light_ctx_cat = _encode_light_context_features(
            tab=tab,
            hour_bucket=hour_bucket,
            date_bucket=date_bucket,
            short_history_len=short_history_counts,
            short_strong_count=short_strong_counts,
            long_strong_history_len=long_strong_counts,
        )

        candidate_ids_per_query: List[np.ndarray] = []
        candidate_positive_mask_per_query: List[np.ndarray] = []
        max_candidates = 0
        for sample_idx, sample in enumerate(batch):
            positive_ids = self._sample_without_replacement(
                np.asarray(sample["positive_video_ids"], dtype=np.int64),
                self.max_positive_items,
                rng,
            )
            explicit_negative_ids = self._sample_without_replacement(
                np.asarray(sample["explicit_negative_video_ids"], dtype=np.int64),
                self.num_explicit_negatives,
                rng,
            )
            history_ids = short_history_video_ids[sample_idx]
            exclude_ids = set(int(v) for v in positive_ids.tolist())
            exclude_ids.update(int(v) for v in history_ids.tolist())

            ordered_ids: List[int] = []
            ordered_labels: List[float] = []
            ordered_set: set[int] = set()
            for item_id in positive_ids.tolist():
                if int(item_id) not in ordered_set:
                    ordered_ids.append(int(item_id))
                    ordered_labels.append(1.0)
                    ordered_set.add(int(item_id))

            for item_id in explicit_negative_ids.tolist():
                item_id = int(item_id)
                if item_id in exclude_ids or item_id in ordered_set:
                    continue
                ordered_ids.append(item_id)
                ordered_labels.append(0.0)
                ordered_set.add(item_id)
                exclude_ids.add(item_id)

            hard_negative_ids = self.hard_negative_sampler.sample_hard_negatives(
                positive_video_ids=positive_ids.tolist(),
                exclude_ids=exclude_ids,
                num_samples=self.num_hard_negatives,
                rng=rng,
            )
            for item_id in hard_negative_ids:
                if item_id in ordered_set:
                    continue
                ordered_ids.append(int(item_id))
                ordered_labels.append(0.0)
                ordered_set.add(int(item_id))
                exclude_ids.add(int(item_id))

            random_negative_ids = self.hard_negative_sampler.sample_random_negatives(
                exclude_ids=exclude_ids,
                num_samples=self.num_random_negatives,
                rng=rng,
            )
            for item_id in random_negative_ids:
                if item_id in ordered_set:
                    continue
                ordered_ids.append(int(item_id))
                ordered_labels.append(0.0)
                ordered_set.add(int(item_id))
                exclude_ids.add(int(item_id))

            if len(ordered_ids) > 1:
                order = rng.permutation(len(ordered_ids))
                candidate_ids = np.asarray([ordered_ids[i] for i in order], dtype=np.int64)
                positive_mask = np.asarray([ordered_labels[i] for i in order], dtype=np.float32)
            else:
                candidate_ids = np.asarray(ordered_ids, dtype=np.int64)
                positive_mask = np.asarray(ordered_labels, dtype=np.float32)

            candidate_ids_per_query.append(candidate_ids)
            candidate_positive_mask_per_query.append(positive_mask)
            max_candidates = max(max_candidates, len(candidate_ids))

        candidate_item_emb = np.zeros((len(batch), max_candidates, self.content_dim), dtype=np.float32)
        candidate_mask = np.zeros((len(batch), max_candidates), dtype=np.float32)
        candidate_positive_mask = np.zeros((len(batch), max_candidates), dtype=np.float32)
        for i, candidate_ids in enumerate(candidate_ids_per_query):
            if len(candidate_ids) == 0:
                continue
            item_idx, valid = self.content_store.lookup_indices(candidate_ids)
            if not np.any(valid):
                continue
            take = int(valid.sum())
            candidate_item_emb[i, :take] = self.content_store.item_vecs[item_idx[valid]]
            candidate_mask[i, :take] = 1.0
            candidate_positive_mask[i, :take] = candidate_positive_mask_per_query[i][valid]

        return {
            "light_ctx_cat": torch.from_numpy(light_ctx_cat),
            "hist_content_emb": torch.from_numpy(hist_content_emb),
            "hist_weights": torch.from_numpy(hist_weights),
            "hist_mask": torch.from_numpy(hist_mask),
            "neg_hist_content_emb": torch.from_numpy(neg_hist_content_emb),
            "neg_hist_weights": torch.from_numpy(neg_hist_weights),
            "neg_hist_mask": torch.from_numpy(neg_hist_mask),
            "neg_history_stats": torch.from_numpy(neg_history_stats),
            "long_term_pref": torch.from_numpy(long_term_pref),
            "candidate_item_emb": torch.from_numpy(candidate_item_emb),
            "candidate_mask": torch.from_numpy(candidate_mask),
            "candidate_positive_mask": torch.from_numpy(candidate_positive_mask),
        }


def _build_candidate_item_bank(
    model: ContentTwoTowerRecallModel,
    content_store: ContentEmbeddingStore,
    device: torch.device,
    batch_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    item_ids = content_store.item_ids
    item_vecs_np = content_store.item_vecs

    model.eval()
    all_vecs: List[torch.Tensor] = []
    with torch.no_grad():
        starts = range(0, len(item_ids), batch_size)
        iterator = tqdm(
            starts,
            total=(len(item_ids) + batch_size - 1) // batch_size if len(item_ids) > 0 else 0,
            desc="Content Candidate Encode",
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for start in iterator:
            end = min(len(item_ids), start + batch_size)
            emb = torch.from_numpy(item_vecs_np[start:end]).to(device)
            vec = model.encode_item(item_content_emb=emb).detach()
            all_vecs.append(vec)
    out_vecs = (
        torch.cat(all_vecs, dim=0)
        if all_vecs
        else torch.zeros((0, model.item_mlp[-1].out_features), device=device)
    )
    return item_ids, out_vecs


def _build_eval_query_features(
    samples: Sequence[EvalUserSample],
    content_store: ContentEmbeddingStore,
    category_store: CategoryFeatureStore,
    short_history_len: int,
    long_term_history_len: int,
    negative_history_len: int,
    short_signal_weight_map: Dict[int, float],
    long_signal_weight_map: Dict[int, float],
    negative_signal_weight_map: Dict[int, float],
    negative_semantic_similarity_threshold: float,
    short_half_life_ms: float,
    negative_half_life_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    content_dim = int(content_store.item_vecs.shape[1]) if content_store.item_vecs.ndim == 2 else 0
    long_term_dim = int(category_store.num_category_l1 + category_store.num_category_l2)
    hist_content_emb = np.zeros((len(samples), short_history_len, content_dim), dtype=np.float32)
    hist_weights = np.zeros((len(samples), short_history_len), dtype=np.float32)
    hist_mask = np.zeros((len(samples), short_history_len), dtype=np.float32)
    neg_hist_content_emb = np.zeros((len(samples), negative_history_len, content_dim), dtype=np.float32)
    neg_hist_weights = np.zeros((len(samples), negative_history_len), dtype=np.float32)
    neg_hist_mask = np.zeros((len(samples), negative_history_len), dtype=np.float32)
    neg_history_stats = np.zeros((len(samples), 2), dtype=np.float32)
    long_term_pref = np.zeros((len(samples), long_term_dim), dtype=np.float32)
    short_history_counts = np.zeros((len(samples),), dtype=np.int64)
    short_strong_counts = np.zeros((len(samples),), dtype=np.int64)
    long_strong_counts = np.zeros((len(samples),), dtype=np.int64)

    for i, sample in enumerate(samples):
        (
            hist_content_emb[i],
            hist_weights[i],
            hist_mask[i],
            _,
            short_history_counts[i],
            short_strong_counts[i],
        ) = _build_short_term_features(
            raw_item_ids=sample.content_history_video_ids,
            raw_signal_types=sample.content_history_signal_types,
            raw_event_times=sample.content_history_time_ms,
            query_time_ms=sample.time_ms,
            content_store=content_store,
            short_history_len=short_history_len,
            signal_weight_map=short_signal_weight_map,
            half_life_ms=short_half_life_ms,
        )
        (
            neg_hist_content_emb[i],
            neg_hist_weights[i],
            neg_hist_mask[i],
            neg_count,
            neg_recentness,
        ) = _build_negative_history_features(
            raw_item_ids=sample.negative_history_video_ids,
            raw_signal_types=sample.negative_history_signal_types,
            raw_event_times=sample.negative_history_time_ms,
            query_time_ms=sample.time_ms,
            content_store=content_store,
            negative_history_len=negative_history_len,
            half_life_ms=negative_half_life_ms,
            signal_weight_map=negative_signal_weight_map,
            semantic_similarity_threshold=negative_semantic_similarity_threshold,
            reference_hist_emb=hist_content_emb[i],
            reference_hist_weights=hist_weights[i],
            reference_hist_mask=hist_mask[i],
        )
        neg_history_stats[i, 0] = (
            float(neg_count) / float(max(negative_history_len, 1)) if negative_history_len > 0 else 0.0
        )
        neg_history_stats[i, 1] = float(neg_recentness)
        long_term_pref[i], long_strong_counts[i] = _build_long_term_preference(
            raw_item_ids=sample.strong_history_video_ids,
            raw_signal_types=sample.strong_history_signal_types,
            category_store=category_store,
            long_term_history_len=long_term_history_len,
            signal_weight_map=long_signal_weight_map,
        )

    tab = np.asarray([sample.tab for sample in samples], dtype=np.int64)
    hour = np.asarray([sample.hour_bucket for sample in samples], dtype=np.int64)
    date = np.asarray([sample.date_bucket for sample in samples], dtype=np.int64)
    light_ctx_cat = _encode_light_context_features(
        tab=tab,
        hour_bucket=hour,
        date_bucket=date,
        short_history_len=short_history_counts,
        short_strong_count=short_strong_counts,
        long_strong_history_len=long_strong_counts,
    )
    return (
        light_ctx_cat,
        hist_content_emb,
        hist_weights,
        hist_mask,
        long_term_pref,
        neg_hist_content_emb,
        neg_hist_weights,
        neg_hist_mask,
        neg_history_stats,
    )


def evaluate_content_recall(
    model: ContentTwoTowerRecallModel,
    content_store: ContentEmbeddingStore,
    category_store: CategoryFeatureStore,
    eval_samples: List[EvalUserSample],
    topk: List[int],
    short_history_len: int,
    long_term_history_len: int,
    negative_history_len: int,
    short_signal_weight_map: Dict[int, float],
    long_signal_weight_map: Dict[int, float],
    negative_signal_weight_map: Dict[int, float],
    negative_semantic_similarity_threshold: float,
    short_half_life_ms: float,
    negative_half_life_ms: float,
    device: torch.device,
    recall_topn: int,
    item_batch_size: int,
    query_batch_size: int,
    skip_no_candidate_positive: bool,
    show_progress: bool,
) -> Dict[str, float]:
    if not eval_samples:
        out = {f"hr@{k}": 0.0 for k in topk}
        out.update({f"ndcg@{k}": 0.0 for k in topk})
        out["num_users"] = 0.0
        out["num_eval_points"] = 0.0
        out["num_unique_users"] = 0.0
        out["num_skipped"] = 0.0
        out["avg_recall_candidates"] = 0.0
        out.update(
            compute_content_immunity_metrics(
                pred_lists=[],
                samples=[],
                category_store=category_store,
                topk=topk,
                negative_history_len=negative_history_len,
            )
        )
        return out

    cand_item_ids, cand_item_vecs = _build_candidate_item_bank(
        model=model,
        content_store=content_store,
        device=device,
        batch_size=item_batch_size,
        show_progress=show_progress,
    )
    cand_item_set = set(cand_item_ids.tolist())
    max_k = max(max(topk), recall_topn)
    totals = {f"hr@{k}": 0.0 for k in topk}
    totals.update({f"ndcg@{k}": 0.0 for k in topk})
    skipped = 0
    candidate_total = 0.0

    active_samples: List[EvalUserSample] = []
    for sample in eval_samples:
        gt = sample.positives
        if skip_no_candidate_positive and len(gt.intersection(cand_item_set)) == 0:
            skipped += 1
            continue
        active_samples.append(sample)

    if not active_samples:
        out = {k: 0.0 for k in totals}
        out["num_users"] = 0.0
        out["num_eval_points"] = 0.0
        out["num_unique_users"] = 0.0
        out["num_skipped"] = float(skipped)
        out["num_candidates"] = float(len(cand_item_ids))
        out["avg_recall_candidates"] = 0.0
        out.update(
            compute_content_immunity_metrics(
                pred_lists=[],
                samples=[],
                category_store=category_store,
                topk=topk,
                negative_history_len=negative_history_len,
            )
        )
        return out

    (
        light_ctx_cat,
        hist_content_emb,
        hist_weights,
        hist_mask,
        long_term_pref,
        neg_hist_content_emb,
        neg_hist_weights,
        neg_hist_mask,
        neg_history_stats,
    ) = _build_eval_query_features(
        samples=active_samples,
        content_store=content_store,
        category_store=category_store,
        short_history_len=short_history_len,
        long_term_history_len=long_term_history_len,
        negative_history_len=negative_history_len,
        short_signal_weight_map=short_signal_weight_map,
        long_signal_weight_map=long_signal_weight_map,
        negative_signal_weight_map=negative_signal_weight_map,
        negative_semantic_similarity_threshold=negative_semantic_similarity_threshold,
        short_half_life_ms=short_half_life_ms,
        negative_half_life_ms=negative_half_life_ms,
    )

    model.eval()
    with torch.no_grad():
        user_vecs = model.encode_user(
            light_ctx_cat=torch.from_numpy(light_ctx_cat).to(device),
            hist_content_emb=torch.from_numpy(hist_content_emb).to(device),
            hist_weights=torch.from_numpy(hist_weights).to(device),
            hist_mask=torch.from_numpy(hist_mask).to(device),
            long_term_pref=torch.from_numpy(long_term_pref).to(device),
            neg_hist_content_emb=torch.from_numpy(neg_hist_content_emb).to(device),
            neg_hist_weights=torch.from_numpy(neg_hist_weights).to(device),
            neg_hist_mask=torch.from_numpy(neg_hist_mask).to(device),
            neg_history_stats=torch.from_numpy(neg_history_stats).to(device),
        )
        top_items = batched_topk_inner_product_search(
            query_vecs=user_vecs,
            item_vecs=cand_item_vecs,
            item_ids=cand_item_ids,
            topk=max_k,
            query_batch_size=query_batch_size,
            show_progress=show_progress,
            progress_desc="Content Recall Search",
        )

    used = len(active_samples)
    unique_users = len({sample.user_id for sample in active_samples})
    for sample, pred_items in zip(active_samples, top_items.tolist()):
        gt = sample.positives
        pred = list(pred_items[:recall_topn])
        candidate_total += float(len(pred))
        for k in topk:
            totals[f"hr@{k}"] += hit_rate_at_k(pred, gt_items=gt, k=k)
            totals[f"ndcg@{k}"] += ndcg_at_k(pred, gt_items=gt, k=k)

    out = {k: (v / used if used > 0 else 0.0) for k, v in totals.items()}
    out["num_users"] = float(used)
    out["num_eval_points"] = float(used)
    out["num_unique_users"] = float(unique_users)
    out["num_skipped"] = float(skipped)
    out["num_candidates"] = float(len(cand_item_ids))
    out["avg_recall_candidates"] = float(candidate_total / used if used > 0 else 0.0)
    pred_lists = [list(row[:recall_topn]) for row in top_items.tolist()]
    out.update(
        compute_content_immunity_metrics(
            pred_lists=pred_lists,
            samples=active_samples,
            category_store=category_store,
            topk=topk,
            negative_history_len=negative_history_len,
        )
    )
    return out


def train_one_epoch(
    model: ContentTwoTowerRecallModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    temperature: float,
    device: torch.device,
    amp_enabled: bool,
    show_progress: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_queries = 0
    total_avg_candidates = 0.0

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=f"Content Recall Epoch {epoch}",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in iterator:
        batch = _move_batch_to_device(batch, device=device)
        candidate_mask = batch["candidate_mask"] > 0.5
        if candidate_mask.sum().item() == 0:
            continue

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            user_vec = model.encode_user(
                light_ctx_cat=batch["light_ctx_cat"],
                hist_content_emb=batch["hist_content_emb"],
                hist_weights=batch["hist_weights"],
                hist_mask=batch["hist_mask"],
                long_term_pref=batch["long_term_pref"],
                neg_hist_content_emb=batch["neg_hist_content_emb"],
                neg_hist_weights=batch["neg_hist_weights"],
                neg_hist_mask=batch["neg_hist_mask"],
                neg_history_stats=batch["neg_history_stats"],
            )
            item_vec = model.encode_item(item_content_emb=batch["candidate_item_emb"])
            logits = torch.einsum("bd,bkd->bk", user_vec, item_vec) / temperature
            loss, active_queries, avg_candidates = _multi_positive_sampled_softmax_loss(
                logits=logits,
                candidate_mask=batch["candidate_mask"],
                positive_mask=batch["candidate_positive_mask"],
            )

        if active_queries == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        total_batches += 1
        total_queries += int(active_queries)
        total_avg_candidates += float(avg_candidates)
        if show_progress:
            iterator.set_postfix(loss=f"{(total_loss / max(total_batches, 1)):.4f}")

    return {
        "loss": total_loss / max(total_batches, 1),
        "batches": float(total_batches),
        "active_queries": float(total_queries),
        "avg_candidates": total_avg_candidates / max(total_batches, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    topk = parse_topk(args.topk)
    if not topk:
        raise ValueError("topk is empty")

    if args.short_history_len <= 0:
        raise ValueError("short_history_len must be > 0")

    processed_dir = args.processed_dir
    train_path = processed_dir / "interactions.train.csv"
    valid_path = processed_dir / "interactions.valid.csv"
    item_path = processed_dir / "item_features.selected.csv"

    short_signal_weight_map = {
        1: float(args.click_weight),
        2: float(args.long_view_weight),
        3: float(args.like_weight),
        4: float(args.social_weight),
    }
    long_signal_weight_map = {
        2: float(args.long_view_weight),
        3: float(args.like_weight),
        4: float(args.social_weight),
    }
    negative_signal_weight_map = {
        1: float(args.negative_relative_weight),
        2: float(args.negative_hate_weight),
    }
    short_half_life_ms = max(float(args.short_history_half_life_hours), 0.0) * 3600.0 * 1000.0
    negative_half_life_ms = max(float(args.negative_half_life_hours), 0.0) * 3600.0 * 1000.0

    print("[ContentRecall] loading interactions ...")
    train_df = load_interactions(
        csv_path=train_path,
        max_rows=args.train_max_rows,
        sample_frac=1.0,
        positive_only_mode=None,
        seed=args.seed,
    )
    valid_df = load_interactions(
        csv_path=valid_path,
        max_rows=args.valid_max_rows,
        sample_frac=1.0,
        positive_only_mode=None,
        seed=args.seed,
    )
    print(f"[ContentRecall] train rows={len(train_df)}, valid rows={len(valid_df)}")

    print("[ContentRecall] loading content / category assets ...")
    raw_train_item_ids = np.unique(train_df["video_id"].to_numpy(dtype=np.int64))
    train_content_store = load_content_embedding_store(
        embedding_path=args.content_item_emb,
        video_id_to_index_path=args.content_video_id_to_index,
        candidate_item_ids=raw_train_item_ids,
    )
    eval_content_store = load_content_embedding_store(
        embedding_path=args.content_item_emb,
        video_id_to_index_path=args.content_video_id_to_index,
        candidate_item_ids=None,
    )
    category_store = load_category_feature_store(
        category_asset_path=args.content_category_csv,
        raw_category_csv=args.raw_category_csv,
        candidate_item_ids=eval_content_store.item_ids,
    )
    train_item_store = ItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=dict(DEFAULT_BUCKET_SIZES),
        candidate_video_ids=train_content_store.item_ids,
    )
    print(
        f"[ContentRecall] train content items={len(train_content_store.item_ids)}, "
        f"eval content items={len(eval_content_store.item_ids)}, "
        f"category_l1={category_store.num_category_l1}, "
        f"category_l2={category_store.num_category_l2}"
    )

    item_counts = train_df["video_id"].value_counts(sort=False)
    sampler_item_ids = item_counts.index.to_numpy(dtype=np.int64, copy=False)
    sampler_item_counts = item_counts.to_numpy(dtype=np.float64, copy=False)
    hard_negative_sampler = RecallHardNegativeSampler(
        item_store=train_item_store,
        candidate_item_ids=sampler_item_ids,
        candidate_item_counts=sampler_item_counts,
    )

    train_ds = ContentRecallTrainQueryDataset(
        frame=train_df,
        positive_label_mode=args.label_mode,
        negative_history_len=args.negative_history_len,
        sample_frac=args.train_sample_frac,
        seed=args.seed,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No content recall training queries were built from train_df.")
    print(
        f"[ContentRecall] train queries={len(train_ds)} "
        f"avg_pos={train_ds.avg_positive_items:.2f} "
        f"avg_explicit_neg={train_ds.avg_explicit_negative_items:.2f} "
        f"avg_neg_hist={train_ds.avg_negative_history_items:.2f}"
    )

    collator = ContentRecallQueryBatchCollator(
        content_store=train_content_store,
        category_store=category_store,
        short_history_len=args.short_history_len,
        long_term_history_len=args.long_term_history_len,
        negative_history_len=args.negative_history_len,
        short_signal_weight_map=short_signal_weight_map,
        long_signal_weight_map=long_signal_weight_map,
        negative_signal_weight_map=negative_signal_weight_map,
        negative_semantic_similarity_threshold=args.negative_semantic_sim_threshold,
        short_half_life_hours=args.short_history_half_life_hours,
        negative_half_life_hours=args.negative_half_life_hours,
        hard_negative_sampler=hard_negative_sampler,
        num_explicit_negatives=args.num_explicit_negatives,
        num_hard_negatives=args.num_hard_negatives,
        num_random_negatives=args.num_random_negatives,
        max_positive_items=args.max_positive_items_per_query,
        seed=args.seed,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, **loader_kwargs)

    eval_samples = build_pointwise_eval_samples(
        interactions=valid_df,
        positive_label_mode=args.label_mode,
        max_samples=args.max_eval_users,
        max_negative_history_len=max(200, int(args.negative_history_len)),
    )
    print(f"[ContentRecall] eval points={len(eval_samples)}")

    content_dim = int(train_content_store.item_vecs.shape[1])
    long_term_dim = int(category_store.num_category_l1 + category_store.num_category_l2)
    model = ContentTwoTowerRecallModel(
        bucket_sizes=LIGHT_CONTEXT_BUCKET_SIZES,
        content_dim=content_dim,
        embedding_dim=args.embedding_dim,
        tower_dim=args.tower_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        long_term_input_dim=long_term_dim,
        negative_immunity_mode=args.negative_immunity_mode,
        negative_fixed_alpha=args.negative_fixed_alpha,
        negative_max_alpha=args.negative_max_alpha,
        negative_gate_hidden_dim=args.negative_gate_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history = []
    best_score = -1.0
    best_metrics = {}
    best_path = output_dir / "content_recall_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            temperature=args.temperature,
            device=device,
            amp_enabled=amp_enabled,
            show_progress=not args.disable_progress,
            epoch=epoch,
        )
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        valid_metrics = {}
        train_elapsed = time.time() - t0
        valid_elapsed = 0.0
        if should_eval:
            t1 = time.time()
            valid_metrics = evaluate_content_recall(
                model=model,
                content_store=eval_content_store,
                category_store=category_store,
                eval_samples=eval_samples,
                topk=topk,
                short_history_len=args.short_history_len,
                long_term_history_len=args.long_term_history_len,
                negative_history_len=args.negative_history_len,
                short_signal_weight_map=short_signal_weight_map,
                long_signal_weight_map=long_signal_weight_map,
                negative_signal_weight_map=negative_signal_weight_map,
                negative_semantic_similarity_threshold=args.negative_semantic_sim_threshold,
                short_half_life_ms=short_half_life_ms,
                negative_half_life_ms=negative_half_life_ms,
                device=device,
                recall_topn=args.recall_topn,
                item_batch_size=args.eval_item_batch_size,
                query_batch_size=args.eval_query_batch_size,
                skip_no_candidate_positive=args.skip_no_candidate_positive,
                show_progress=not args.disable_progress,
            )
            valid_elapsed = time.time() - t1

        main_k = topk[-1]
        score = valid_metrics.get(f"hr@{main_k}", best_score)
        history.append(
            {
                "epoch": epoch,
                "train_elapsed_sec": train_elapsed,
                "valid_elapsed_sec": valid_elapsed,
                "train": train_stats,
                "valid": valid_metrics,
            }
        )
        if should_eval:
            print(
                f"[ContentRecall][Epoch {epoch}] "
                f"loss={train_stats['loss']:.5f} "
                f"hr@{main_k}={score:.5f} "
                f"ndcg@{main_k}={valid_metrics[f'ndcg@{main_k}']:.5f} "
                f"train={train_elapsed:.1f}s valid={valid_elapsed:.1f}s"
            )
        else:
            print(
                f"[ContentRecall][Epoch {epoch}] "
                f"loss={train_stats['loss']:.5f} "
                f"train={train_elapsed:.1f}s valid=skipped"
            )

        if should_eval and score > best_score:
            best_score = score
            best_metrics = valid_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "light_context_bucket_sizes": LIGHT_CONTEXT_BUCKET_SIZES,
                    "config": vars(args),
                    "content_dim": content_dim,
                    "long_term_input_dim": long_term_dim,
                },
                best_path,
            )

    write_ids_to_npy(output_dir / "candidate_item_ids.npy", eval_content_store.item_ids)
    write_ids_to_npy(output_dir / "train_candidate_item_ids.npy", train_content_store.item_ids)

    summary = {
        "best_valid_score": best_score,
        "best_valid_metrics": best_metrics,
        "epochs": history,
        "train_rows": int(len(train_df)),
        "train_queries": int(len(train_ds)),
        "valid_rows": int(len(valid_df)),
        "num_train_items": int(len(train_content_store.item_ids)),
        "num_eval_items": int(len(eval_content_store.item_ids)),
        "content_dim": int(content_dim),
        "long_term_input_dim": int(long_term_dim),
        "num_category_l1": int(category_store.num_category_l1),
        "num_category_l2": int(category_store.num_category_l2),
        "avg_positive_items_per_query": float(train_ds.avg_positive_items),
        "avg_explicit_negative_items_per_query": float(train_ds.avg_explicit_negative_items),
        "avg_negative_history_items_per_query": float(train_ds.avg_negative_history_items),
        "light_context_features": list(LIGHT_CONTEXT_BUCKET_SIZES.keys()),
    }
    save_json(output_dir / "content_recall_train_summary.json", summary)
    print(f"[ContentRecall] best checkpoint: {best_path}")
    print(f"[ContentRecall] summary saved: {output_dir / 'content_recall_train_summary.json'}")


if __name__ == "__main__":
    main()
