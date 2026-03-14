#!/usr/bin/env python3
"""Shared data utilities for KuaiRand two-stage baseline."""

from __future__ import annotations

import csv
import json
import math
import random
import warnings
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

def _patch_torch_pytree_compat() -> None:
    """
    Some environments combine older torch with newer transformers.
    Transformers expects torch.utils._pytree.register_pytree_node, while
    older torch versions only expose _register_pytree_node.
    """
    try:
        import torch.utils._pytree as _torch_pytree
    except Exception:
        return

    if hasattr(_torch_pytree, "register_pytree_node"):
        return
    if not hasattr(_torch_pytree, "_register_pytree_node"):
        return

    def _compat_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        to_str_fn=None,
        maybe_from_str_fn=None,
        *,
        serialized_type_name=None,
        to_dumpable_context=None,
        from_dumpable_context=None,
        flatten_with_keys_fn=None,
    ):
        del to_str_fn, maybe_from_str_fn, serialized_type_name, flatten_with_keys_fn
        return _torch_pytree._register_pytree_node(
            cls,
            flatten_fn,
            unflatten_fn,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )

    _torch_pytree.register_pytree_node = _compat_register_pytree_node


_patch_torch_pytree_compat()

"""
类别特征做 embedding
"""
USER_CATEGORICAL_COLUMNS = [
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
]

"""
数值特征 归一化输入MLP
"""
USER_NUMERIC_COLUMNS = [
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
]
"""
物品侧特征 也分为离散特征和数值特征
"""
ITEM_CATEGORICAL_COLUMNS = [
    "video_id",
    "author_id",
    "video_type",
    "upload_type",
    "music_id",
    "music_type",
    "tag",
]

ITEM_NUMERIC_COLUMNS = [
    "video_duration",
    "server_width",
    "server_height",
]

"""
上下文特征
"""
CONTEXT_CATEGORICAL_COLUMNS = [
    "tab",
    "hour_bucket",
    "date_bucket",
]

INTERACTION_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "tab",
    "hour_bucket",
    "date_bucket",
    "history_pos_video_ids",
    "history_click_video_ids",
    "history_click_len",
    "content_history_video_ids",
    "content_history_signal_types",
    "content_history_time_ms",
    "content_history_len",
    "strong_history_video_ids",
    "strong_history_signal_types",
    "strong_history_time_ms",
    "strong_history_len",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "time_ms",
]

"""
每个离散特征映射到多大的桶空间中 也就是将离散特征的分桶量
"""
DEFAULT_BUCKET_SIZES = {
    # Keep a generous user bucket so both 1K and Pure variants avoid heavy collisions.
    "user_id": 65_536,
    "user_active_degree": 32,
    "is_lowactive_period": 8,
    "is_live_streamer": 8,
    "is_video_author": 8,
    "onehot_feat0": 8,
    "onehot_feat1": 32,
    "onehot_feat2": 256,
    "onehot_feat3": 4096,
    "onehot_feat4": 64,
    "onehot_feat5": 128,
    "onehot_feat6": 16,
    "onehot_feat7": 512,
    "onehot_feat8": 2048,
    "onehot_feat9": 32,
    "onehot_feat10": 32,
    "onehot_feat11": 64,
    "onehot_feat12": 64,
    "onehot_feat13": 64,
    "onehot_feat14": 64,
    "onehot_feat15": 64,
    "onehot_feat16": 64,
    "onehot_feat17": 64,
    "video_id": 1_000_003,
    "author_id": 500_003,
    "video_type": 128,
    "upload_type": 128,
    "music_id": 500_003,
    "music_type": 1024,
    "tag": 2048,
    "tab": 64,
    "hour_bucket": 64,
    "date_bucket": 128,
}


RANK_USER_CATEGORICAL_COLUMNS = [
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
]

RANK_USER_NUMERIC_COLUMNS = [
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
]

RANK_USER_RANGE_FALLBACK_COLUMNS = {
    "follow_user_num_range": "follow_user_num",
    "fans_user_num_range": "fans_user_num",
    "friend_user_num_range": "friend_user_num",
    "register_days_range": "register_days",
}

RANK_USER_STRING_COLUMNS = {
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
}

RANK_ITEM_CATEGORICAL_COLUMNS = [
    "video_id",
    "author_id",
    "video_type",
    "upload_type",
    "visible_status",
    "music_id",
    "music_type",
    "tag",
]

RANK_ITEM_STRING_COLUMNS = {
    "video_type",
    "upload_type",
}

RANK_ITEM_NUMERIC_COLUMNS = [
    "video_duration",
    "server_width",
    "server_height",
]

RANK_CONTEXT_CATEGORICAL_COLUMNS = [
    "tab",
    "hour_bucket",
    "weekday",
    "is_weekend",
    "is_night",
]

RANK_CONTEXT_NUMERIC_COLUMNS = [
    "time_frac",
]

DEFAULT_RANK_BUCKET_SIZES = {
    "user_id": 65_536,
    "user_active_degree": 32,
    "is_lowactive_period": 8,
    "is_live_streamer": 8,
    "is_video_author": 8,
    "follow_user_num_range": 128,
    "fans_user_num_range": 128,
    "friend_user_num_range": 128,
    "register_days_range": 128,
    "onehot_feat0": 8,
    "onehot_feat1": 32,
    "onehot_feat2": 256,
    "onehot_feat3": 4096,
    "onehot_feat4": 64,
    "onehot_feat5": 128,
    "onehot_feat6": 16,
    "onehot_feat7": 512,
    "onehot_feat8": 2048,
    "onehot_feat9": 32,
    "onehot_feat10": 32,
    "onehot_feat11": 64,
    "onehot_feat12": 64,
    "onehot_feat13": 64,
    "onehot_feat14": 64,
    "onehot_feat15": 64,
    "onehot_feat16": 64,
    "onehot_feat17": 64,
    "video_id": 1_000_003,
    "author_id": 500_003,
    "video_type": 128,
    "upload_type": 128,
    "visible_status": 16,
    "music_id": 500_003,
    "music_type": 1024,
    "tag": 2048,
    "tab": 32,
    "hour_bucket": 32,
    "weekday": 16,
    "is_weekend": 8,
    "is_night": 8,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_cuda = torch.cuda.is_available()
        if has_cuda:
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _hash_str_to_bucket(value: str, bucket_size: int, prefix: str) -> int:
    """
    字符串特征映射到桶中
    """
    if bucket_size <= 1:
        return 0
    msg = f"{prefix}::{value}".encode("utf-8")
    h = zlib.crc32(msg) & 0xFFFFFFFF
    return 1 + (h % (bucket_size - 1))


def bucketize_int_array(values: np.ndarray, bucket_size: int) -> np.ndarray:
    """
    数值特种映射到桶中
    """
    if bucket_size <= 1:
        return np.zeros_like(values, dtype=np.int64)
    out = np.abs(values.astype(np.int64))
    out = (out % (bucket_size - 1)) + 1
    return out.astype(np.int64)


def bucketize_str_series(values: pd.Series, bucket_size: int, prefix: str) -> np.ndarray:
    text_values = values.fillna("").astype(str).tolist()
    return np.array(
        [_hash_str_to_bucket(v, bucket_size=bucket_size, prefix=prefix) for v in text_values],
        dtype=np.int64,
    )


def _safe_numeric(values: np.ndarray) -> np.ndarray:
    """
    空值、负值处理 并且进行log1p操作 防止长尾影响
    """
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    return np.log1p(arr)


def _normalize_with_stats(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    标准化
    """
    denom = np.where(std > 1e-6, std, 1.0)
    return ((values - mean) / denom).astype(np.float32)


def positive_mask(df: pd.DataFrame, label_mode: str) -> pd.Series:
    """
    正样本定义
    """
    click = pd.to_numeric(df["is_click"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_like = pd.to_numeric(df["is_like"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_follow = pd.to_numeric(df["is_follow"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_comment = pd.to_numeric(df["is_comment"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_forward = pd.to_numeric(df["is_forward"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    is_hate = pd.to_numeric(df["is_hate"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    long_view = pd.to_numeric(df["long_view"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    if label_mode == "click":
        mask = click == 1
        return pd.Series(mask, index=df.index, dtype=bool)
    if label_mode == "long_view":
        mask = long_view == 1
        return pd.Series(mask, index=df.index, dtype=bool)
    if label_mode == "click_or_long":
        mask = (click == 1) | (long_view == 1)
        return pd.Series(mask, index=df.index, dtype=bool)
    if label_mode == "signal_positive":
        mask = (
            ((is_follow == 1) | (is_comment == 1) | (is_forward == 1))
            | (is_like == 1)
            | (long_view == 1)
            | ((click == 1) & (is_hate == 0))
        )
        return pd.Series(mask, index=df.index, dtype=bool)
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def load_interactions(
    csv_path: Path,
    max_rows: Optional[int] = None,
    sample_frac: float = 1.0,
    positive_only_mode: Optional[str] = None,
    seed: int = 42,
    usecols: Optional[List[str]] = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    requested_usecols = list(usecols or INTERACTION_COLUMNS)
    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [col for col in requested_usecols if col in header_cols]
    if not usecols:
        return pd.DataFrame(columns=requested_usecols)
    frames: List[pd.DataFrame] = []
    total = 0
    rng = np.random.default_rng(seed)

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        if positive_only_mode is not None:
            chunk = chunk[positive_mask(chunk, positive_only_mode)]
        if sample_frac < 1.0:
            if len(chunk) == 0:
                continue
            take = rng.random(len(chunk)) < sample_frac
            chunk = chunk[take]
        if len(chunk) == 0:
            continue

        if max_rows is not None and total + len(chunk) > max_rows:
            need = max_rows - total
            if need <= 0:
                break
            chunk = chunk.iloc[:need]
        frames.append(chunk)
        total += len(chunk)
        if max_rows is not None and total >= max_rows:
            break

    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, axis=0, ignore_index=True)


def collect_video_ids(csv_paths: Sequence[Path], max_rows_each: Optional[int] = None) -> np.ndarray:
    """
    只抽取训练、验证测试中珍重用到的物品的特征
    """
    video_ids: List[np.ndarray] = []
    for path in csv_paths:
        df = load_interactions(
            csv_path=path,
            max_rows=max_rows_each,
            sample_frac=1.0,
            usecols=["video_id"],
            positive_only_mode=None,
        )
        if len(df) > 0:
            video_ids.append(df["video_id"].to_numpy(dtype=np.int64))
    if not video_ids:
        return np.zeros((0,), dtype=np.int64)
    out = np.unique(np.concatenate(video_ids, axis=0))
    return out

@dataclass
class UserFeatureStore:
    """
        读取用户数据、缺失值填充、类别特征分桶、数值特征标准化 并按照用户排序
    """
    user_ids: np.ndarray
    cat_features: np.ndarray
    num_features: np.ndarray
    bucket_sizes: Dict[str, int]
    num_mean: np.ndarray
    num_std: np.ndarray

    @classmethod
    def from_csv(cls, csv_path: Path, bucket_sizes: Dict[str, int]) -> "UserFeatureStore":
        need_cols = list(dict.fromkeys(USER_CATEGORICAL_COLUMNS + USER_NUMERIC_COLUMNS))
        df = pd.read_csv(csv_path, usecols=need_cols)
        df = df.fillna(0)

        cat_parts: List[np.ndarray] = []
        for col in USER_CATEGORICAL_COLUMNS:
            bucket = bucket_sizes[col]
            if col in {"user_active_degree"}:
                cat = bucketize_str_series(df[col], bucket, prefix=col)
            else:
                values = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
                cat = bucketize_int_array(values, bucket)
            cat_parts.append(cat[:, None])
        cat_arr = np.concatenate(cat_parts, axis=1).astype(np.int64)

        num_raw = df[USER_NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
        num_log = _safe_numeric(num_raw)
        mean = num_log.mean(axis=0, keepdims=True)
        std = num_log.std(axis=0, keepdims=True)
        num_arr = _normalize_with_stats(num_log, mean, std).astype(np.float32)

        user_ids = pd.to_numeric(df["user_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        order = np.argsort(user_ids)
        user_ids = user_ids[order]
        cat_arr = cat_arr[order]
        num_arr = num_arr[order]

        return cls(
            user_ids=user_ids,
            cat_features=cat_arr,
            num_features=num_arr,
            bucket_sizes=dict(bucket_sizes),
            num_mean=mean.reshape(-1).astype(np.float32),
            num_std=std.reshape(-1).astype(np.float32),
        )

    def lookup(self, user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(self.user_ids, user_ids)
        valid = (idx < len(self.user_ids)) & (self.user_ids[np.clip(idx, 0, len(self.user_ids) - 1)] == user_ids)

        cat = np.zeros((len(user_ids), self.cat_features.shape[1]), dtype=np.int64)
        num = np.zeros((len(user_ids), self.num_features.shape[1]), dtype=np.float32)

        if len(user_ids) > 0:
            user_bucket = self.bucket_sizes["user_id"]
            cat[:, 0] = bucketize_int_array(user_ids.astype(np.int64), user_bucket)

        if np.any(valid):
            valid_idx = idx[valid]
            cat[valid] = self.cat_features[valid_idx]
            num[valid] = self.num_features[valid_idx]
        return cat, num

    def to_metadata(self) -> dict:
        return {
            "num_mean": self.num_mean.tolist(),
            "num_std": self.num_std.tolist(),
        }

    @property
    def num_users(self) -> int:
        return int(len(self.user_ids))


@dataclass
class ItemFeatureStore:
    """
        读取物品特征 按照video排序
    """
    item_ids: np.ndarray
    cat_features: np.ndarray
    num_features: np.ndarray
    bucket_sizes: Dict[str, int]
    num_mean: np.ndarray
    num_std: np.ndarray

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        bucket_sizes: Dict[str, int],
        candidate_video_ids: Optional[np.ndarray] = None,
        max_items: Optional[int] = None,
        chunksize: int = 250_000,
    ) -> "ItemFeatureStore":
        candidate_set = set(candidate_video_ids.tolist()) if candidate_video_ids is not None else None
        cols = list(dict.fromkeys(ITEM_CATEGORICAL_COLUMNS + ITEM_NUMERIC_COLUMNS))

        item_id_parts: List[np.ndarray] = []
        cat_parts: List[np.ndarray] = []
        num_parts: List[np.ndarray] = []
        loaded = 0

        for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize):
            chunk = chunk.fillna(0)
            if candidate_set is not None:
                chunk = chunk[chunk["video_id"].isin(candidate_set)]
            if len(chunk) == 0:
                continue

            if max_items is not None and loaded >= max_items:
                break
            if max_items is not None and loaded + len(chunk) > max_items:
                need = max_items - loaded
                if need <= 0:
                    break
                chunk = chunk.iloc[:need]

            video_id = pd.to_numeric(chunk["video_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            author_id = pd.to_numeric(chunk["author_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            music_id = pd.to_numeric(chunk["music_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            music_type = pd.to_numeric(chunk["music_type"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            tag = pd.to_numeric(chunk["tag"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

            cat = np.zeros((len(chunk), len(ITEM_CATEGORICAL_COLUMNS)), dtype=np.int64)
            cat[:, 0] = bucketize_int_array(video_id, bucket_sizes["video_id"])
            cat[:, 1] = bucketize_int_array(author_id, bucket_sizes["author_id"])
            cat[:, 2] = bucketize_str_series(chunk["video_type"], bucket_sizes["video_type"], "video_type")
            cat[:, 3] = bucketize_str_series(chunk["upload_type"], bucket_sizes["upload_type"], "upload_type")
            cat[:, 4] = bucketize_int_array(music_id, bucket_sizes["music_id"])
            cat[:, 5] = bucketize_int_array(music_type, bucket_sizes["music_type"])
            cat[:, 6] = bucketize_int_array(tag, bucket_sizes["tag"])

            num_raw = chunk[ITEM_NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
            num_log = _safe_numeric(num_raw)

            item_id_parts.append(video_id.astype(np.int64))
            cat_parts.append(cat)
            num_parts.append(num_log.astype(np.float32))
            loaded += len(chunk)

        if not item_id_parts:
            return cls(
                item_ids=np.zeros((0,), dtype=np.int64),
                cat_features=np.zeros((0, len(ITEM_CATEGORICAL_COLUMNS)), dtype=np.int64),
                num_features=np.zeros((0, len(ITEM_NUMERIC_COLUMNS)), dtype=np.float32),
                bucket_sizes=dict(bucket_sizes),
                num_mean=np.zeros((len(ITEM_NUMERIC_COLUMNS),), dtype=np.float32),
                num_std=np.ones((len(ITEM_NUMERIC_COLUMNS),), dtype=np.float32),
            )

        item_ids = np.concatenate(item_id_parts, axis=0)
        cat_arr = np.concatenate(cat_parts, axis=0)
        num_arr = np.concatenate(num_parts, axis=0)

        order = np.argsort(item_ids)
        item_ids = item_ids[order]
        cat_arr = cat_arr[order]
        num_arr = num_arr[order]

        # Deduplicate by video_id, keep first after sorting.
        if len(item_ids) > 1:
            keep = np.ones(len(item_ids), dtype=bool)
            keep[1:] = item_ids[1:] != item_ids[:-1]
            item_ids = item_ids[keep]
            cat_arr = cat_arr[keep]
            num_arr = num_arr[keep]

        mean = num_arr.mean(axis=0, keepdims=True)
        std = num_arr.std(axis=0, keepdims=True)
        num_arr = _normalize_with_stats(num_arr, mean, std)

        return cls(
            item_ids=item_ids.astype(np.int64),
            cat_features=cat_arr.astype(np.int64),
            num_features=num_arr.astype(np.float32),
            bucket_sizes=dict(bucket_sizes),
            num_mean=mean.reshape(-1).astype(np.float32),
            num_std=std.reshape(-1).astype(np.float32),
        )

    def lookup(self, video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.item_ids) == 0:
            cat = np.zeros((len(video_ids), len(ITEM_CATEGORICAL_COLUMNS)), dtype=np.int64)
            num = np.zeros((len(video_ids), len(ITEM_NUMERIC_COLUMNS)), dtype=np.float32)
            valid = np.zeros((len(video_ids),), dtype=bool)
            if len(video_ids) > 0:
                cat[:, 0] = bucketize_int_array(video_ids.astype(np.int64), self.bucket_sizes["video_id"])
            return cat, num, valid

        idx = np.searchsorted(self.item_ids, video_ids)
        idx_clip = np.clip(idx, 0, len(self.item_ids) - 1)
        valid = (idx < len(self.item_ids)) & (self.item_ids[idx_clip] == video_ids)

        cat = np.zeros((len(video_ids), self.cat_features.shape[1]), dtype=np.int64)
        num = np.zeros((len(video_ids), self.num_features.shape[1]), dtype=np.float32)

        if len(video_ids) > 0:
            cat[:, 0] = bucketize_int_array(video_ids.astype(np.int64), self.bucket_sizes["video_id"])

        if np.any(valid):
            v_idx = idx[valid]
            cat[valid] = self.cat_features[v_idx]
            num[valid] = self.num_features[v_idx]
        return cat, num, valid

    def lookup_indices(self, video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        video_ids = np.asarray(video_ids, dtype=np.int64)
        if len(self.item_ids) == 0:
            return np.zeros((len(video_ids),), dtype=np.int64), np.zeros((len(video_ids),), dtype=bool)
        idx = np.searchsorted(self.item_ids, video_ids)
        idx_clip = np.clip(idx, 0, len(self.item_ids) - 1)
        valid = (idx < len(self.item_ids)) & (self.item_ids[idx_clip] == video_ids)
        out_idx = np.zeros((len(video_ids),), dtype=np.int64)
        if np.any(valid):
            out_idx[valid] = idx[valid]
        return out_idx, valid

    def iter_feature_batches(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        n = len(self.item_ids)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            yield (
                self.item_ids[start:end],
                self.cat_features[start:end],
                self.num_features[start:end],
            )

    @property
    def num_items(self) -> int:
        return int(len(self.item_ids))

    def to_metadata(self) -> dict:
        return {
            "num_mean": self.num_mean.tolist(),
            "num_std": self.num_std.tolist(),
        }


@dataclass
class RankUserFeatureStore:
    user_ids: np.ndarray
    cat_features: np.ndarray
    num_features: np.ndarray
    bucket_sizes: Dict[str, int]
    num_mean: np.ndarray
    num_std: np.ndarray

    @classmethod
    def from_csv(cls, csv_path: Path, bucket_sizes: Dict[str, int]) -> "RankUserFeatureStore":
        need_cols = ["user_id"] + list(RANK_USER_CATEGORICAL_COLUMNS) + list(RANK_USER_NUMERIC_COLUMNS)
        need_cols.extend(RANK_USER_RANGE_FALLBACK_COLUMNS.values())
        header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        usecols = [col for col in dict.fromkeys(need_cols) if col in header_cols]
        df = pd.read_csv(csv_path, usecols=usecols).fillna(0)
        if "user_id" not in df.columns:
            raise ValueError(f"user_id is missing from {csv_path}")

        cat_parts: List[np.ndarray] = []
        for col in RANK_USER_CATEGORICAL_COLUMNS:
            bucket = int(bucket_sizes[col])
            if col in df.columns:
                if col in RANK_USER_STRING_COLUMNS:
                    cat = bucketize_str_series(df[col], bucket_size=bucket, prefix=col)
                else:
                    values = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
                    cat = bucketize_int_array(values, bucket)
            else:
                fallback_col = RANK_USER_RANGE_FALLBACK_COLUMNS.get(col)
                if fallback_col is not None and fallback_col in df.columns:
                    values = pd.to_numeric(df[fallback_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
                    cat = bucketize_int_array(values, bucket)
                else:
                    cat = np.zeros((len(df),), dtype=np.int64)
            cat_parts.append(cat[:, None])

        user_ids = pd.to_numeric(df["user_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        cat_arr = np.concatenate(cat_parts, axis=1).astype(np.int64)
        num_raw = np.zeros((len(df), len(RANK_USER_NUMERIC_COLUMNS)), dtype=np.float32)
        for idx, col in enumerate(RANK_USER_NUMERIC_COLUMNS):
            if col in df.columns:
                num_raw[:, idx] = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        num_log = _safe_numeric(num_raw)
        mean = num_log.mean(axis=0, keepdims=True) if len(df) > 0 else np.zeros((1, len(RANK_USER_NUMERIC_COLUMNS)), dtype=np.float32)
        std = num_log.std(axis=0, keepdims=True) if len(df) > 0 else np.ones((1, len(RANK_USER_NUMERIC_COLUMNS)), dtype=np.float32)
        num_arr = _normalize_with_stats(num_log, mean, std).astype(np.float32)
        order = np.argsort(user_ids, kind="mergesort")
        return cls(
            user_ids=user_ids[order],
            cat_features=cat_arr[order],
            num_features=num_arr[order],
            bucket_sizes=dict(bucket_sizes),
            num_mean=mean.reshape(-1).astype(np.float32),
            num_std=std.reshape(-1).astype(np.float32),
        )

    def lookup(self, user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        if len(self.user_ids) == 0:
            cat = np.zeros((len(user_ids), len(RANK_USER_CATEGORICAL_COLUMNS)), dtype=np.int64)
            num = np.zeros((len(user_ids), len(RANK_USER_NUMERIC_COLUMNS)), dtype=np.float32)
            if len(user_ids) > 0:
                cat[:, 0] = bucketize_int_array(user_ids, self.bucket_sizes["user_id"])
            return cat, num
        idx = np.searchsorted(self.user_ids, user_ids)
        idx_clip = np.clip(idx, 0, max(len(self.user_ids) - 1, 0))
        valid = (idx < len(self.user_ids)) & (self.user_ids[idx_clip] == user_ids)

        cat = np.zeros((len(user_ids), self.cat_features.shape[1]), dtype=np.int64)
        num = np.zeros((len(user_ids), self.num_features.shape[1]), dtype=np.float32)
        if len(user_ids) > 0:
            cat[:, 0] = bucketize_int_array(user_ids, self.bucket_sizes["user_id"])
        if np.any(valid):
            cat[valid] = self.cat_features[idx[valid]]
            num[valid] = self.num_features[idx[valid]]
        return cat, num


@dataclass
class RankItemFeatureStore:
    item_ids: np.ndarray
    cat_features: np.ndarray
    num_features: np.ndarray
    bucket_sizes: Dict[str, int]
    num_mean: np.ndarray
    num_std: np.ndarray

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        bucket_sizes: Dict[str, int],
        candidate_video_ids: Optional[np.ndarray] = None,
        max_items: Optional[int] = None,
        chunksize: int = 250_000,
    ) -> "RankItemFeatureStore":
        header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        need_cols = ["video_id"] + list(RANK_ITEM_CATEGORICAL_COLUMNS) + list(RANK_ITEM_NUMERIC_COLUMNS)
        usecols = [col for col in dict.fromkeys(need_cols) if col in header_cols]
        candidate_set = set(candidate_video_ids.tolist()) if candidate_video_ids is not None else None

        item_id_parts: List[np.ndarray] = []
        cat_parts: List[np.ndarray] = []
        num_parts: List[np.ndarray] = []
        loaded = 0

        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
            chunk = chunk.fillna(0)
            if candidate_set is not None:
                chunk = chunk[chunk["video_id"].isin(candidate_set)]
            if len(chunk) == 0:
                continue

            if max_items is not None and loaded >= max_items:
                break
            if max_items is not None and loaded + len(chunk) > max_items:
                need = max_items - loaded
                if need <= 0:
                    break
                chunk = chunk.iloc[:need]

            video_id = pd.to_numeric(chunk["video_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            cat = np.zeros((len(chunk), len(RANK_ITEM_CATEGORICAL_COLUMNS)), dtype=np.int64)
            for idx, col in enumerate(RANK_ITEM_CATEGORICAL_COLUMNS):
                bucket = int(bucket_sizes[col])
                if col == "video_id":
                    cat[:, idx] = bucketize_int_array(video_id, bucket)
                    continue
                if col not in chunk.columns:
                    continue
                if col in RANK_ITEM_STRING_COLUMNS:
                    cat[:, idx] = bucketize_str_series(chunk[col], bucket_size=bucket, prefix=col)
                else:
                    values = pd.to_numeric(chunk[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
                    cat[:, idx] = bucketize_int_array(values, bucket)

            num_raw = np.zeros((len(chunk), len(RANK_ITEM_NUMERIC_COLUMNS)), dtype=np.float32)
            for idx, col in enumerate(RANK_ITEM_NUMERIC_COLUMNS):
                if col in chunk.columns:
                    num_raw[:, idx] = pd.to_numeric(chunk[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            num_log = _safe_numeric(num_raw)

            item_id_parts.append(video_id.astype(np.int64))
            cat_parts.append(cat.astype(np.int64))
            num_parts.append(num_log.astype(np.float32))
            loaded += len(chunk)

        if not item_id_parts:
            return cls(
                item_ids=np.zeros((0,), dtype=np.int64),
                cat_features=np.zeros((0, len(RANK_ITEM_CATEGORICAL_COLUMNS)), dtype=np.int64),
                num_features=np.zeros((0, len(RANK_ITEM_NUMERIC_COLUMNS)), dtype=np.float32),
                bucket_sizes=dict(bucket_sizes),
                num_mean=np.zeros((len(RANK_ITEM_NUMERIC_COLUMNS),), dtype=np.float32),
                num_std=np.ones((len(RANK_ITEM_NUMERIC_COLUMNS),), dtype=np.float32),
            )

        item_ids = np.concatenate(item_id_parts, axis=0)
        cat_arr = np.concatenate(cat_parts, axis=0)
        num_arr = np.concatenate(num_parts, axis=0)

        order = np.argsort(item_ids, kind="mergesort")
        item_ids = item_ids[order]
        cat_arr = cat_arr[order]
        num_arr = num_arr[order]

        if len(item_ids) > 1:
            keep = np.ones(len(item_ids), dtype=bool)
            keep[1:] = item_ids[1:] != item_ids[:-1]
            item_ids = item_ids[keep]
            cat_arr = cat_arr[keep]
            num_arr = num_arr[keep]

        mean = num_arr.mean(axis=0, keepdims=True) if len(num_arr) > 0 else np.zeros((1, num_arr.shape[1]), dtype=np.float32)
        std = num_arr.std(axis=0, keepdims=True) if len(num_arr) > 0 else np.ones((1, num_arr.shape[1]), dtype=np.float32)
        num_arr = _normalize_with_stats(num_arr, mean, std)

        return cls(
            item_ids=item_ids.astype(np.int64),
            cat_features=cat_arr.astype(np.int64),
            num_features=num_arr.astype(np.float32),
            bucket_sizes=dict(bucket_sizes),
            num_mean=mean.reshape(-1).astype(np.float32),
            num_std=std.reshape(-1).astype(np.float32),
        )

    def lookup(self, video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        video_ids = np.asarray(video_ids, dtype=np.int64)
        if len(self.item_ids) == 0:
            cat = np.zeros((len(video_ids), len(RANK_ITEM_CATEGORICAL_COLUMNS)), dtype=np.int64)
            num = np.zeros((len(video_ids), len(RANK_ITEM_NUMERIC_COLUMNS)), dtype=np.float32)
            valid = np.zeros((len(video_ids),), dtype=bool)
            if len(video_ids) > 0:
                cat[:, 0] = bucketize_int_array(video_ids, self.bucket_sizes["video_id"])
            return cat, num, valid

        idx = np.searchsorted(self.item_ids, video_ids)
        idx_clip = np.clip(idx, 0, len(self.item_ids) - 1)
        valid = (idx < len(self.item_ids)) & (self.item_ids[idx_clip] == video_ids)

        cat = np.zeros((len(video_ids), self.cat_features.shape[1]), dtype=np.int64)
        num = np.zeros((len(video_ids), self.num_features.shape[1]), dtype=np.float32)
        if len(video_ids) > 0:
            cat[:, 0] = bucketize_int_array(video_ids, self.bucket_sizes["video_id"])
        if np.any(valid):
            v_idx = idx[valid]
            cat[valid] = self.cat_features[v_idx]
            num[valid] = self.num_features[v_idx]
        return cat, num, valid

    @property
    def num_items(self) -> int:
        return int(len(self.item_ids))


def encode_context_features(
    tab: np.ndarray,
    hour_bucket: np.ndarray,
    date_bucket: np.ndarray,
    bucket_sizes: Dict[str, int],
) -> np.ndarray:
    """
        将交互上下文特征离散化为类别id
    """
    out = np.zeros((len(tab), len(CONTEXT_CATEGORICAL_COLUMNS)), dtype=np.int64)
    out[:, 0] = bucketize_int_array(tab.astype(np.int64), bucket_sizes["tab"])
    out[:, 1] = bucketize_int_array(hour_bucket.astype(np.int64), bucket_sizes["hour_bucket"])
    out[:, 2] = bucketize_int_array(date_bucket.astype(np.int64), bucket_sizes["date_bucket"])
    return out


def encode_rank_context_features(
    tab: np.ndarray,
    date: np.ndarray,
    hourmin: np.ndarray,
    time_ms: np.ndarray,
    bucket_sizes: Dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    cat = np.zeros((len(tab), len(RANK_CONTEXT_CATEGORICAL_COLUMNS)), dtype=np.int64)
    tab_arr = np.asarray(tab, dtype=np.int64)
    date_arr = np.asarray(date, dtype=np.int64)
    hourmin_arr = np.asarray(hourmin, dtype=np.int64)
    hour_bucket = hourmin_arr // 100
    weekday = np.zeros((len(tab_arr),), dtype=np.int64)
    if len(tab_arr) > 0:
        parsed = pd.to_datetime(date_arr.astype(str), format="%Y%m%d", errors="coerce")
        weekday = np.nan_to_num(parsed.dt.weekday.to_numpy(dtype=np.float32), nan=0.0).astype(np.int64)
    is_weekend = (weekday >= 5).astype(np.int64)
    is_night = ((hour_bucket <= 5) | (hour_bucket >= 23)).astype(np.int64)

    cat[:, 0] = bucketize_int_array(tab_arr, bucket_sizes["tab"])
    cat[:, 1] = bucketize_int_array(hour_bucket, bucket_sizes["hour_bucket"])
    cat[:, 2] = bucketize_int_array(weekday, bucket_sizes["weekday"])
    cat[:, 3] = bucketize_int_array(is_weekend, bucket_sizes["is_weekend"])
    cat[:, 4] = bucketize_int_array(is_night, bucket_sizes["is_night"])

    num = np.zeros((len(tab), len(RANK_CONTEXT_NUMERIC_COLUMNS)), dtype=np.float32)
    if len(tab) > 0:
        time_arr = np.asarray(time_ms, dtype=np.int64)
        num[:, 0] = (time_arr % 86_400_000).astype(np.float32) / 86_400_000.0
    return cat, num


def parse_history_field(raw: str, max_history_len: int) -> List[int]:
    """
    将History解析为整数列表
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if text == "" or text == "nan":
        return []
    items = [x for x in text.split(",") if x]
    if max_history_len > 0 and len(items) > max_history_len:
        items = items[-max_history_len:]
    out: List[int] = []
    for token in items:
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def encode_history_batch(
    raw_histories: Sequence[str],
    max_history_len: int,
    item_bucket_size: int,
    return_raw_ids: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.zeros((len(raw_histories), max_history_len), dtype=np.int64)
    mask = np.zeros((len(raw_histories), max_history_len), dtype=np.float32)
    raw_ids = (
        np.zeros((len(raw_histories), max_history_len), dtype=np.int64) if return_raw_ids else None
    )
    for i, raw in enumerate(raw_histories):
        hist = parse_history_field(raw, max_history_len=max_history_len)
        if not hist:
            continue
        h = np.array(hist, dtype=np.int64)
        if raw_ids is not None:
            raw_ids[i, : len(h)] = h
        h = bucketize_int_array(h, item_bucket_size)
        ids[i, : len(h)] = h
        mask[i, : len(h)] = 1.0
    if raw_ids is not None:
        return ids, mask, raw_ids
    return ids, mask


class BaseInteractionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.user_ids = frame["user_id"].to_numpy(dtype=np.int64)
        self.video_ids = frame["video_id"].to_numpy(dtype=np.int64)
        self.date = (
            frame["date"].to_numpy(dtype=np.int64)
            if "date" in frame.columns
            else np.zeros((len(frame),), dtype=np.int64)
        )
        self.hourmin = (
            frame["hourmin"].to_numpy(dtype=np.int64)
            if "hourmin" in frame.columns
            else np.zeros((len(frame),), dtype=np.int64)
        )
        self.time_ms = (
            frame["time_ms"].to_numpy(dtype=np.int64)
            if "time_ms" in frame.columns
            else np.zeros((len(frame),), dtype=np.int64)
        )
        self.tab = frame["tab"].to_numpy(dtype=np.int64)
        self.hour_bucket = frame["hour_bucket"].to_numpy(dtype=np.int64)
        self.date_bucket = frame["date_bucket"].to_numpy(dtype=np.int64)
        self.history = frame["history_pos_video_ids"].astype(str).tolist()
        self.click_history = (
            frame["history_click_video_ids"].astype(str).tolist()
            if "history_click_video_ids" in frame.columns
            # Older processed files only contain the positive history.
            else self.history
        )
        self.click = frame["is_click"].to_numpy(dtype=np.float32)
        self.like = (
            frame["is_like"].to_numpy(dtype=np.float32)
            if "is_like" in frame.columns
            else np.zeros((len(frame),), dtype=np.float32)
        )
        self.long_view = frame["long_view"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.user_ids)

"""
召回和排序所需的特征 排序相比召回多了监督标签
"""
class RecallTrainDataset(BaseInteractionDataset):
    def __getitem__(self, idx: int) -> dict:
        return {
            "user_id": int(self.user_ids[idx]),
            "video_id": int(self.video_ids[idx]),
            "tab": int(self.tab[idx]),
            "hour_bucket": int(self.hour_bucket[idx]),
            "date_bucket": int(self.date_bucket[idx]),
            "history": self.history[idx],
        }


@dataclass
class RecallTrainQuerySample:
    user_id: int
    tab: int
    hour_bucket: int
    date_bucket: int
    history: str
    positive_video_ids: np.ndarray
    explicit_negative_video_ids: np.ndarray


class RecallTrainQueryDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        positive_label_mode: str,
        sample_frac: float = 1.0,
        seed: int = 42,
    ):
        query_frame = frame.copy()
        # Keep original intra-timestep order so the first row still carries the
        # pre-step history even when older processed files are reused.
        query_frame = query_frame.sort_values(["user_id", "time_ms"], kind="mergesort")
        pos = positive_mask(query_frame, positive_label_mode)
        query_frame["is_positive"] = pos.astype(np.int8)

        samples: List[RecallTrainQuerySample] = []
        for (_, _), group in query_frame.groupby(["user_id", "time_ms"], sort=False):
            positives = np.unique(
                group.loc[group["is_positive"] == 1, "video_id"].to_numpy(dtype=np.int64)
            )
            if len(positives) == 0:
                continue
            negatives = np.unique(
                group.loc[group["is_positive"] == 0, "video_id"].to_numpy(dtype=np.int64)
            )
            first = group.iloc[0]
            samples.append(
                RecallTrainQuerySample(
                    user_id=int(first["user_id"]),
                    tab=int(first["tab"]),
                    hour_bucket=int(first["hour_bucket"]),
                    date_bucket=int(first["date_bucket"]),
                    history=str(first["history_pos_video_ids"]),
                    positive_video_ids=positives,
                    explicit_negative_video_ids=negatives,
                )
            )

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "user_id": sample.user_id,
            "tab": sample.tab,
            "hour_bucket": sample.hour_bucket,
            "date_bucket": sample.date_bucket,
            "history": sample.history,
            "positive_video_ids": sample.positive_video_ids,
            "explicit_negative_video_ids": sample.explicit_negative_video_ids,
        }


class RecallTrainQueryCacheDataset(Dataset):
    CACHE_VERSION = 2

    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.user_cat = arrays["user_cat"]
        self.user_num = arrays["user_num"]
        self.ctx_cat = arrays["ctx_cat"]
        self.history_bucket_values = arrays["history_bucket_values"]
        self.history_bucket_offsets = arrays["history_bucket_offsets"]
        self.history_item_index_values = arrays["history_item_index_values"]
        self.history_item_index_offsets = arrays["history_item_index_offsets"]
        self.positive_item_index_values = arrays["positive_item_index_values"]
        self.positive_item_index_offsets = arrays["positive_item_index_offsets"]
        self.explicit_negative_item_index_values = arrays["explicit_negative_item_index_values"]
        self.explicit_negative_item_index_offsets = arrays["explicit_negative_item_index_offsets"]
        self.hard_negative_item_index_values = arrays["hard_negative_item_index_values"]
        self.hard_negative_item_index_offsets = arrays["hard_negative_item_index_offsets"]
        self.random_negative_item_index_values = arrays["random_negative_item_index_values"]
        self.random_negative_item_index_offsets = arrays["random_negative_item_index_offsets"]
        self.avg_positive_items = float(arrays["avg_positive_items"][0]) if len(arrays["avg_positive_items"]) > 0 else 0.0
        self.avg_explicit_negative_items = (
            float(arrays["avg_explicit_negative_items"][0])
            if len(arrays["avg_explicit_negative_items"]) > 0
            else 0.0
        )

    @classmethod
    def build_or_load(
        cls,
        cache_path: Path,
        frame: pd.DataFrame,
        positive_label_mode: str,
        user_store: UserFeatureStore,
        item_store: ItemFeatureStore,
        bucket_sizes: Dict[str, int],
        max_history_len: int,
        hard_negative_sampler: RecallHardNegativeSampler,
        hard_negative_cache_size: int,
        random_negative_cache_size: int,
        sample_frac: float = 1.0,
        seed: int = 42,
        rebuild: bool = False,
    ) -> "RecallTrainQueryCacheDataset":
        if cache_path.exists() and not rebuild:
            with np.load(cache_path, allow_pickle=False, mmap_mode="r") as arrays:
                version = int(arrays["cache_version"][0]) if "cache_version" in arrays else 0
                if version == cls.CACHE_VERSION:
                    return cls({key: arrays[key] for key in arrays.files})

        base_ds = RecallTrainQueryDataset(
            frame=frame,
            positive_label_mode=positive_label_mode,
            sample_frac=sample_frac,
            seed=seed,
        )
        samples = base_ds.samples
        user_ids = np.asarray([sample.user_id for sample in samples], dtype=np.int64)
        tab = np.asarray([sample.tab for sample in samples], dtype=np.int64)
        hour_bucket = np.asarray([sample.hour_bucket for sample in samples], dtype=np.int64)
        date_bucket = np.asarray([sample.date_bucket for sample in samples], dtype=np.int64)
        user_cat, user_num = user_store.lookup(user_ids)
        ctx_cat = encode_context_features(tab, hour_bucket, date_bucket, bucket_sizes)

        history_bucket_sequences: List[np.ndarray] = []
        history_item_index_sequences: List[np.ndarray] = []
        positive_item_index_sequences: List[np.ndarray] = []
        explicit_negative_item_index_sequences: List[np.ndarray] = []
        hard_negative_item_index_sequences: List[np.ndarray] = []
        random_negative_item_index_sequences: List[np.ndarray] = []
        rng = np.random.default_rng(seed)
        for sample in samples:
            history = np.asarray(
                parse_history_field(sample.history, max_history_len=max_history_len),
                dtype=np.int64,
            )
            positives = np.asarray(sample.positive_video_ids, dtype=np.int64)
            explicit_negatives = np.asarray(sample.explicit_negative_video_ids, dtype=np.int64)
            exclude_ids = set(int(v) for v in history.tolist())
            exclude_ids.update(int(v) for v in positives.tolist())
            hard_negatives = hard_negative_sampler.sample_hard_negatives(
                positive_video_ids=positives.tolist(),
                exclude_ids=exclude_ids,
                num_samples=hard_negative_cache_size,
                rng=rng,
            )
            exclude_ids.update(int(v) for v in hard_negatives)
            random_negatives = hard_negative_sampler.sample_random_negatives(
                exclude_ids=exclude_ids,
                num_samples=random_negative_cache_size,
                rng=rng,
            )

            history_bucket_sequences.append(
                bucketize_int_array(history, bucket_sizes["video_id"]).astype(np.int64, copy=False)
            )

            history_item_idx, history_valid = item_store.lookup_indices(history)
            if np.any(history_valid):
                history_item_index_sequences.append(history_item_idx[history_valid].astype(np.int64, copy=False))
            else:
                history_item_index_sequences.append(np.zeros((0,), dtype=np.int64))

            positive_item_idx, positive_valid = item_store.lookup_indices(positives)
            positive_item_index_sequences.append(
                positive_item_idx[positive_valid].astype(np.int64, copy=False)
                if np.any(positive_valid)
                else np.zeros((0,), dtype=np.int64)
            )

            explicit_item_idx, explicit_valid = item_store.lookup_indices(explicit_negatives)
            explicit_negative_item_index_sequences.append(
                explicit_item_idx[explicit_valid].astype(np.int64, copy=False)
                if np.any(explicit_valid)
                else np.zeros((0,), dtype=np.int64)
            )

            hard_negatives_arr = np.asarray(hard_negatives, dtype=np.int64)
            hard_item_idx, hard_valid = item_store.lookup_indices(hard_negatives_arr)
            hard_negative_item_index_sequences.append(
                hard_item_idx[hard_valid].astype(np.int64, copy=False)
                if np.any(hard_valid)
                else np.zeros((0,), dtype=np.int64)
            )

            random_negatives_arr = np.asarray(random_negatives, dtype=np.int64)
            random_item_idx, random_valid = item_store.lookup_indices(random_negatives_arr)
            random_negative_item_index_sequences.append(
                random_item_idx[random_valid].astype(np.int64, copy=False)
                if np.any(random_valid)
                else np.zeros((0,), dtype=np.int64)
            )

        history_bucket_values, history_bucket_offsets = _pack_int_sequences(history_bucket_sequences)
        history_item_index_values, history_item_index_offsets = _pack_int_sequences(history_item_index_sequences)
        positive_item_index_values, positive_item_index_offsets = _pack_int_sequences(positive_item_index_sequences)
        explicit_negative_item_index_values, explicit_negative_item_index_offsets = _pack_int_sequences(
            explicit_negative_item_index_sequences
        )
        hard_negative_item_index_values, hard_negative_item_index_offsets = _pack_int_sequences(
            hard_negative_item_index_sequences
        )
        random_negative_item_index_values, random_negative_item_index_offsets = _pack_int_sequences(
            random_negative_item_index_sequences
        )

        arrays = {
            "cache_version": np.asarray([cls.CACHE_VERSION], dtype=np.int64),
            "user_cat": user_cat.astype(np.int64, copy=False),
            "user_num": user_num.astype(np.float32, copy=False),
            "ctx_cat": ctx_cat.astype(np.int64, copy=False),
            "history_bucket_values": history_bucket_values.astype(np.int64, copy=False),
            "history_bucket_offsets": history_bucket_offsets.astype(np.int64, copy=False),
            "history_item_index_values": history_item_index_values.astype(np.int64, copy=False),
            "history_item_index_offsets": history_item_index_offsets.astype(np.int64, copy=False),
            "positive_item_index_values": positive_item_index_values.astype(np.int64, copy=False),
            "positive_item_index_offsets": positive_item_index_offsets.astype(np.int64, copy=False),
            "explicit_negative_item_index_values": explicit_negative_item_index_values.astype(np.int64, copy=False),
            "explicit_negative_item_index_offsets": explicit_negative_item_index_offsets.astype(np.int64, copy=False),
            "hard_negative_item_index_values": hard_negative_item_index_values.astype(np.int64, copy=False),
            "hard_negative_item_index_offsets": hard_negative_item_index_offsets.astype(np.int64, copy=False),
            "random_negative_item_index_values": random_negative_item_index_values.astype(np.int64, copy=False),
            "random_negative_item_index_offsets": random_negative_item_index_offsets.astype(np.int64, copy=False),
            "avg_positive_items": np.asarray([base_ds.avg_positive_items], dtype=np.float32),
            "avg_explicit_negative_items": np.asarray(
                [base_ds.avg_explicit_negative_items], dtype=np.float32
            ),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **arrays)
        return cls(arrays)

    def __len__(self) -> int:
        return int(self.user_cat.shape[0])

    @staticmethod
    def _slice(values: np.ndarray, offsets: np.ndarray, idx: int) -> np.ndarray:
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        return np.asarray(values[start:end], dtype=np.int64)

    def __getitem__(self, idx: int) -> dict:
        return {
            "user_cat": np.asarray(self.user_cat[idx], dtype=np.int64),
            "user_num": np.asarray(self.user_num[idx], dtype=np.float32),
            "ctx_cat": np.asarray(self.ctx_cat[idx], dtype=np.int64),
            "history_bucket_ids": self._slice(self.history_bucket_values, self.history_bucket_offsets, idx),
            "history_item_indices": self._slice(
                self.history_item_index_values,
                self.history_item_index_offsets,
                idx,
            ),
            "positive_item_indices": self._slice(
                self.positive_item_index_values,
                self.positive_item_index_offsets,
                idx,
            ),
            "explicit_negative_item_indices": self._slice(
                self.explicit_negative_item_index_values,
                self.explicit_negative_item_index_offsets,
                idx,
            ),
            "hard_negative_item_indices": self._slice(
                self.hard_negative_item_index_values,
                self.hard_negative_item_index_offsets,
                idx,
            ),
            "random_negative_item_indices": self._slice(
                self.random_negative_item_index_values,
                self.random_negative_item_index_offsets,
                idx,
            ),
        }


class RankTrainDataset(BaseInteractionDataset):
    def __getitem__(self, idx: int) -> dict:
        return {
            "user_id": int(self.user_ids[idx]),
            "video_id": int(self.video_ids[idx]),
            "date": int(self.date[idx]),
            "hourmin": int(self.hourmin[idx]),
            "time_ms": int(self.time_ms[idx]),
            "tab": int(self.tab[idx]),
            "history": self.click_history[idx],
            "is_click": float(self.click[idx]),
        }

"""
    将数据转化为batch

"""
class RecallBatchCollator:
    def __init__(
        self,
        user_store: UserFeatureStore,
        item_store: ItemFeatureStore,
        bucket_sizes: Dict[str, int],
        max_history_len: int,
        device: torch.device,
    ) -> None:
        self.user_store = user_store
        self.item_store = item_store
        self.bucket_sizes = bucket_sizes
        self.max_history_len = max_history_len
        self.device = device

    def __call__(self, batch: List[dict]) -> dict:
        user_ids = np.array([x["user_id"] for x in batch], dtype=np.int64)
        video_ids = np.array([x["video_id"] for x in batch], dtype=np.int64)
        tab = np.array([x["tab"] for x in batch], dtype=np.int64)
        hour_bucket = np.array([x["hour_bucket"] for x in batch], dtype=np.int64)
        date_bucket = np.array([x["date_bucket"] for x in batch], dtype=np.int64)
        histories = [x["history"] for x in batch]

        user_cat, user_num = self.user_store.lookup(user_ids)
        item_cat, item_num, valid = self.item_store.lookup(video_ids)
        ctx_cat = encode_context_features(tab, hour_bucket, date_bucket, self.bucket_sizes)
        hist_ids, hist_mask, hist_video_ids = encode_history_batch(
            histories,
            max_history_len=self.max_history_len,
            item_bucket_size=self.bucket_sizes["video_id"],
            return_raw_ids=True,
        )

        return {
            "user_ids": torch.from_numpy(user_ids).to(self.device),
            "user_cat": torch.from_numpy(user_cat).to(self.device),
            "user_num": torch.from_numpy(user_num).to(self.device),
            "item_cat": torch.from_numpy(item_cat).to(self.device),
            "item_num": torch.from_numpy(item_num).to(self.device),
            "ctx_cat": torch.from_numpy(ctx_cat).to(self.device),
            "hist_ids": torch.from_numpy(hist_ids).to(self.device),
            "hist_mask": torch.from_numpy(hist_mask).to(self.device),
            "hist_video_ids": torch.from_numpy(hist_video_ids).to(self.device),
            "valid_item_mask": torch.from_numpy(valid.astype(np.float32)).to(self.device),
            "video_ids": torch.from_numpy(video_ids).to(self.device),
        }


class RankBatchCollator:
    def __init__(
        self,
        user_store: RankUserFeatureStore,
        item_store: RankItemFeatureStore,
        bucket_sizes: Dict[str, int],
        max_history_len: int,
        device: torch.device,
    ) -> None:
        self.user_store = user_store
        self.item_store = item_store
        self.bucket_sizes = bucket_sizes
        self.max_history_len = max_history_len
        self.device = device

    def __call__(self, batch: List[dict]) -> dict:
        user_ids = np.array([x["user_id"] for x in batch], dtype=np.int64)
        video_ids = np.array([x["video_id"] for x in batch], dtype=np.int64)
        date = np.array([x["date"] for x in batch], dtype=np.int64)
        hourmin = np.array([x["hourmin"] for x in batch], dtype=np.int64)
        time_ms = np.array([x["time_ms"] for x in batch], dtype=np.int64)
        tab = np.array([x["tab"] for x in batch], dtype=np.int64)
        histories = [x["history"] for x in batch]

        user_cat, user_num = self.user_store.lookup(user_ids)
        item_cat, item_num, valid = self.item_store.lookup(video_ids)
        ctx_cat, ctx_num = encode_rank_context_features(
            tab=tab,
            date=date,
            hourmin=hourmin,
            time_ms=time_ms,
            bucket_sizes=self.bucket_sizes,
        )
        hist_ids, hist_mask, hist_video_ids = encode_history_batch(
            histories,
            max_history_len=self.max_history_len,
            item_bucket_size=self.bucket_sizes["video_id"],
            return_raw_ids=True,
        )
        hist_author_ids = np.zeros_like(hist_ids)
        hist_tag_ids = np.zeros_like(hist_ids)
        if hist_video_ids.size > 0:
            flat_hist = hist_video_ids.reshape(-1)
            flat_cat, _, flat_valid = self.item_store.lookup(flat_hist)
            flat_author = np.zeros_like(flat_hist, dtype=np.int64)
            flat_tag = np.zeros_like(flat_hist, dtype=np.int64)
            if np.any(flat_valid):
                flat_author[flat_valid] = flat_cat[flat_valid, 1]
                flat_tag[flat_valid] = flat_cat[flat_valid, 7]
            hist_author_ids = flat_author.reshape(hist_video_ids.shape)
            hist_tag_ids = flat_tag.reshape(hist_video_ids.shape)

        out = {
            "user_ids": torch.from_numpy(user_ids).to(self.device),
            "video_ids": torch.from_numpy(video_ids).to(self.device),
            "user_cat": torch.from_numpy(user_cat).to(self.device),
            "user_num": torch.from_numpy(user_num).to(self.device),
            "item_cat": torch.from_numpy(item_cat).to(self.device),
            "item_num": torch.from_numpy(item_num).to(self.device),
            "ctx_cat": torch.from_numpy(ctx_cat).to(self.device),
            "ctx_num": torch.from_numpy(ctx_num).to(self.device),
            "hist_ids": torch.from_numpy(hist_ids).to(self.device),
            "hist_author_ids": torch.from_numpy(hist_author_ids).to(self.device),
            "hist_tag_ids": torch.from_numpy(hist_tag_ids).to(self.device),
            "hist_mask": torch.from_numpy(hist_mask).to(self.device),
            "valid_item_mask": torch.from_numpy(valid.astype(np.float32)).to(self.device),
        }
        click = np.array([x["is_click"] for x in batch], dtype=np.float32)
        out["label_click"] = torch.from_numpy(click).to(self.device)
        return out


class RecallHardNegativeSampler:
    def __init__(
        self,
        item_store: ItemFeatureStore,
        candidate_item_ids: np.ndarray,
        candidate_item_counts: Optional[np.ndarray] = None,
    ) -> None:
        self.item_store = item_store
        candidate_item_ids = np.asarray(candidate_item_ids, dtype=np.int64)
        candidate_item_counts = (
            np.asarray(candidate_item_counts, dtype=np.float64)
            if candidate_item_counts is not None
            else None
        )
        cat, _, valid = item_store.lookup(candidate_item_ids)
        self.candidate_item_ids = candidate_item_ids[valid]
        self.candidate_item_cat = cat[valid]
        if len(self.candidate_item_ids) > 1:
            order = np.argsort(self.candidate_item_ids, kind="mergesort")
            self.candidate_item_ids = self.candidate_item_ids[order]
            self.candidate_item_cat = self.candidate_item_cat[order]
        if candidate_item_counts is None:
            probs = np.ones((len(self.candidate_item_ids),), dtype=np.float64)
        else:
            probs = candidate_item_counts[valid].astype(np.float64)
            if len(self.candidate_item_ids) > 1:
                probs = probs[order]
        probs = np.power(np.clip(probs, a_min=1.0, a_max=None), 0.75).astype(np.float64, copy=False)
        prob_sum = float(probs.sum())
        self.random_probs = probs / prob_sum if prob_sum > 0 else np.full_like(probs, 1.0 / max(len(probs), 1))
        self.author_index = self._build_bucket_index(self.candidate_item_cat[:, 1], self.candidate_item_ids)
        self.video_type_index = self._build_bucket_index(self.candidate_item_cat[:, 2], self.candidate_item_ids)
        self.tag_index = self._build_bucket_index(self.candidate_item_cat[:, 6], self.candidate_item_ids)

    @staticmethod
    def _build_bucket_index(bucket_values: np.ndarray, item_ids: np.ndarray) -> Dict[int, np.ndarray]:
        bucket_to_items: Dict[int, List[int]] = defaultdict(list)
        for bucket, item_id in zip(bucket_values.tolist(), item_ids.tolist()):
            bucket_to_items[int(bucket)].append(int(item_id))
        return {
            bucket: np.asarray(ids, dtype=np.int64)
            for bucket, ids in bucket_to_items.items()
        }

    def lookup_candidate_cat(self, video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        video_ids = np.asarray(video_ids, dtype=np.int64)
        if len(self.candidate_item_ids) == 0:
            cat = np.zeros((len(video_ids), len(ITEM_CATEGORICAL_COLUMNS)), dtype=np.int64)
            valid = np.zeros((len(video_ids),), dtype=bool)
            return cat, valid
        idx = np.searchsorted(self.candidate_item_ids, video_ids)
        idx_clip = np.clip(idx, 0, len(self.candidate_item_ids) - 1)
        valid = (idx < len(self.candidate_item_ids)) & (self.candidate_item_ids[idx_clip] == video_ids)
        cat = np.zeros((len(video_ids), self.candidate_item_cat.shape[1]), dtype=np.int64)
        if np.any(valid):
            cat[valid] = self.candidate_item_cat[idx_clip[valid]]
        return cat, valid

    def sample_hard_negatives(
        self,
        positive_video_ids: Sequence[int],
        exclude_ids: set[int],
        num_samples: int,
        rng: np.random.Generator,
    ) -> List[int]:
        if num_samples <= 0 or len(positive_video_ids) == 0:
            return []

        pos_arr = np.asarray(list(positive_video_ids), dtype=np.int64)
        pos_cat, valid = self.lookup_candidate_cat(pos_arr)
        pools: List[np.ndarray] = []
        for row in pos_cat[valid]:
            pools.append(self.author_index.get(int(row[1]), np.zeros((0,), dtype=np.int64)))
            pools.append(self.video_type_index.get(int(row[2]), np.zeros((0,), dtype=np.int64)))
            pools.append(self.tag_index.get(int(row[6]), np.zeros((0,), dtype=np.int64)))

        out: List[int] = []
        local_seen = set(exclude_ids)
        for pool in pools:
            if len(out) >= num_samples:
                break
            if len(pool) == 0:
                continue
            take = min(len(pool), max(4, num_samples * 2))
            if take >= len(pool):
                chosen = pool
            else:
                chosen = pool[rng.choice(len(pool), size=take, replace=False)]
            for item_id in chosen.tolist():
                item_id = int(item_id)
                if item_id in local_seen:
                    continue
                local_seen.add(item_id)
                out.append(item_id)
                if len(out) >= num_samples:
                    break
        return out[:num_samples]

    def sample_random_negatives(
        self,
        exclude_ids: set[int],
        num_samples: int,
        rng: np.random.Generator,
    ) -> List[int]:
        if num_samples <= 0 or len(self.candidate_item_ids) == 0:
            return []

        out: List[int] = []
        local_seen = set(exclude_ids)
        max_attempts = max(num_samples * 32, 64)
        attempts = 0
        while len(out) < num_samples and attempts < max_attempts:
            need = min(max((num_samples - len(out)) * 4, 8), len(self.candidate_item_ids))
            idx = rng.choice(
                len(self.candidate_item_ids),
                size=need,
                replace=True,
                p=self.random_probs,
            )
            for item_id in self.candidate_item_ids[idx].tolist():
                item_id = int(item_id)
                attempts += 1
                if item_id in local_seen:
                    continue
                local_seen.add(item_id)
                out.append(item_id)
                if len(out) >= num_samples:
                    break
        return out


class RecallQueryBatchCollator:
    def __init__(
        self,
        user_store: UserFeatureStore,
        item_store: ItemFeatureStore,
        bucket_sizes: Dict[str, int],
        max_history_len: int,
        device: torch.device,
        hard_negative_sampler: RecallHardNegativeSampler,
        num_explicit_negatives: int,
        num_hard_negatives: int,
        num_random_negatives: int,
        max_positive_items: int,
        seed: int = 42,
    ) -> None:
        self.user_store = user_store
        self.item_store = item_store
        self.bucket_sizes = bucket_sizes
        self.max_history_len = max_history_len
        self.device = device
        self.hard_negative_sampler = hard_negative_sampler
        self.num_explicit_negatives = max(0, int(num_explicit_negatives))
        self.num_hard_negatives = max(0, int(num_hard_negatives))
        self.num_random_negatives = max(0, int(num_random_negatives))
        self.max_positive_items = max(1, int(max_positive_items))
        self.seed = int(seed)
        self._call_count = 0

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
        use_cached_indices = "positive_item_indices" in batch[0]
        if "user_cat" in batch[0]:
            user_cat = np.stack([np.asarray(x["user_cat"], dtype=np.int64) for x in batch], axis=0)
            user_num = np.stack([np.asarray(x["user_num"], dtype=np.float32) for x in batch], axis=0)
            ctx_cat = np.stack([np.asarray(x["ctx_cat"], dtype=np.int64) for x in batch], axis=0)
            hist_ids = np.zeros((len(batch), self.max_history_len), dtype=np.int64)
            hist_mask = np.zeros((len(batch), self.max_history_len), dtype=np.float32)
            if use_cached_indices:
                hist_item_indices_per_query: List[np.ndarray] = []
            else:
                hist_video_ids = np.zeros((len(batch), self.max_history_len), dtype=np.int64)
            for i, sample in enumerate(batch):
                if use_cached_indices:
                    history = np.asarray(sample["history_bucket_ids"], dtype=np.int64)
                else:
                    history = np.asarray(sample["history_raw_ids"], dtype=np.int64)
                if len(history) == 0:
                    if use_cached_indices:
                        hist_item_indices_per_query.append(np.zeros((0,), dtype=np.int64))
                    continue
                if len(history) > self.max_history_len:
                    history = history[-self.max_history_len :]
                take = len(history)
                hist_ids[i, :take] = history if use_cached_indices else bucketize_int_array(
                    history,
                    self.bucket_sizes["video_id"],
                )
                hist_mask[i, :take] = 1.0
                if use_cached_indices:
                    hist_item_indices = np.asarray(sample["history_item_indices"], dtype=np.int64)
                    if len(hist_item_indices) > self.max_history_len:
                        hist_item_indices = hist_item_indices[-self.max_history_len :]
                    hist_item_indices_per_query.append(hist_item_indices)
                else:
                    hist_video_ids[i, :take] = history
        else:
            user_ids = np.array([x["user_id"] for x in batch], dtype=np.int64)
            tab = np.array([x["tab"] for x in batch], dtype=np.int64)
            hour_bucket = np.array([x["hour_bucket"] for x in batch], dtype=np.int64)
            date_bucket = np.array([x["date_bucket"] for x in batch], dtype=np.int64)
            histories = [x["history"] for x in batch]

            user_cat, user_num = self.user_store.lookup(user_ids)
            ctx_cat = encode_context_features(tab, hour_bucket, date_bucket, self.bucket_sizes)
            hist_ids, hist_mask, hist_video_ids = encode_history_batch(
                histories,
                max_history_len=self.max_history_len,
                item_bucket_size=self.bucket_sizes["video_id"],
                return_raw_ids=True,
            )

        candidate_ids_per_query: List[np.ndarray] = []
        candidate_positive_mask_per_query: List[np.ndarray] = []
        max_candidates = 0

        for sample_idx, sample in enumerate(batch):
            if use_cached_indices:
                positive_ids = self._sample_without_replacement(
                    np.asarray(sample["positive_item_indices"], dtype=np.int64),
                    self.max_positive_items,
                    rng,
                )
                explicit_negative_ids = self._sample_without_replacement(
                    np.asarray(sample["explicit_negative_item_indices"], dtype=np.int64),
                    self.num_explicit_negatives,
                    rng,
                )
                history_ids = hist_item_indices_per_query[sample_idx]
            else:
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
                history_ids = hist_video_ids[sample_idx][hist_mask[sample_idx] > 0.5].astype(np.int64, copy=False)
            exclude_ids = set(int(v) for v in positive_ids.tolist())
            exclude_ids.update(int(v) for v in history_ids.tolist())

            ordered_ids: List[int] = []
            ordered_labels: List[float] = []
            ordered_set: set[int] = set()
            for item_id in positive_ids.tolist():
                item_id = int(item_id)
                if item_id not in ordered_set:
                    ordered_ids.append(item_id)
                    ordered_labels.append(1.0)
                    ordered_set.add(item_id)

            for item_id in explicit_negative_ids.tolist():
                item_id = int(item_id)
                if item_id in exclude_ids or item_id in ordered_set:
                    continue
                ordered_ids.append(item_id)
                ordered_labels.append(0.0)
                ordered_set.add(item_id)
                exclude_ids.add(item_id)

            if "hard_negative_item_indices" in sample:
                hard_negative_ids = self._sample_without_replacement(
                    np.asarray(sample["hard_negative_item_indices"], dtype=np.int64),
                    self.num_hard_negatives,
                    rng,
                ).tolist()
            elif "hard_negative_video_ids" in sample:
                hard_negative_ids = self._sample_without_replacement(
                    np.asarray(sample["hard_negative_video_ids"], dtype=np.int64),
                    self.num_hard_negatives,
                    rng,
                ).tolist()
            else:
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

            if "random_negative_item_indices" in sample:
                random_negative_ids = self._sample_without_replacement(
                    np.asarray(sample["random_negative_item_indices"], dtype=np.int64),
                    self.num_random_negatives,
                    rng,
                ).tolist()
            else:
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

        item_cat_dim = len(ITEM_CATEGORICAL_COLUMNS)
        item_num_dim = len(ITEM_NUMERIC_COLUMNS)
        candidate_item_cat = np.zeros((len(batch), max_candidates, item_cat_dim), dtype=np.int64)
        candidate_item_num = np.zeros((len(batch), max_candidates, item_num_dim), dtype=np.float32)
        candidate_mask = np.zeros((len(batch), max_candidates), dtype=np.float32)
        candidate_positive_mask = np.zeros((len(batch), max_candidates), dtype=np.float32)

        if use_cached_indices:
            for i, candidate_indices in enumerate(candidate_ids_per_query):
                if len(candidate_indices) == 0:
                    continue
                take = len(candidate_indices)
                candidate_item_cat[i, :take] = self.item_store.cat_features[candidate_indices]
                candidate_item_num[i, :take] = self.item_store.num_features[candidate_indices]
                candidate_mask[i, :take] = 1.0
                candidate_positive_mask[i, :take] = candidate_positive_mask_per_query[i]
        else:
            candidate_lengths = np.asarray([len(ids) for ids in candidate_ids_per_query], dtype=np.int64)
            if candidate_lengths.sum() > 0:
                flat_candidate_ids = np.concatenate(candidate_ids_per_query, axis=0)
                unique_candidate_ids, inverse = np.unique(flat_candidate_ids, return_inverse=True)
                unique_item_cat, unique_item_num, unique_valid = self.item_store.lookup(unique_candidate_ids)
                valid_unique_idx = np.flatnonzero(unique_valid)
                valid_item_cat = unique_item_cat[unique_valid]
                valid_item_num = unique_item_num[unique_valid]
                valid_id_map = np.full((len(unique_candidate_ids),), -1, dtype=np.int64)
                valid_id_map[valid_unique_idx] = np.arange(len(valid_unique_idx), dtype=np.int64)
                flat_valid_pos = valid_id_map[inverse]

                offset = 0
                for i, length in enumerate(candidate_lengths.tolist()):
                    if length <= 0:
                        continue
                    end = offset + length
                    query_valid_pos = flat_valid_pos[offset:end]
                    valid_mask = query_valid_pos >= 0
                    if np.any(valid_mask):
                        take = int(valid_mask.sum())
                        take_pos = query_valid_pos[valid_mask]
                        candidate_item_cat[i, :take] = valid_item_cat[take_pos]
                        candidate_item_num[i, :take] = valid_item_num[take_pos]
                        candidate_mask[i, :take] = 1.0
                        candidate_positive_mask[i, :take] = candidate_positive_mask_per_query[i][valid_mask]
                    offset = end

        return {
            "user_cat": torch.from_numpy(user_cat).to(self.device),
            "user_num": torch.from_numpy(user_num).to(self.device),
            "ctx_cat": torch.from_numpy(ctx_cat).to(self.device),
            "hist_ids": torch.from_numpy(hist_ids).to(self.device),
            "hist_mask": torch.from_numpy(hist_mask).to(self.device),
            "candidate_item_cat": torch.from_numpy(candidate_item_cat).to(self.device),
            "candidate_item_num": torch.from_numpy(candidate_item_num).to(self.device),
            "candidate_mask": torch.from_numpy(candidate_mask).to(self.device),
            "candidate_positive_mask": torch.from_numpy(candidate_positive_mask).to(self.device),
        }

"""
评测用户样本构造
"""
@dataclass
class EvalUserSample:
    user_id: int
    tab: int
    date: int
    hourmin: int
    hour_bucket: int
    date_bucket: int
    time_ms: int
    history: str
    click_history: str
    positives: set[int]
    content_history_video_ids: str = ""
    content_history_signal_types: str = ""
    content_history_time_ms: str = ""
    strong_history_video_ids: str = ""
    strong_history_signal_types: str = ""
    strong_history_time_ms: str = ""
    negative_history_video_ids: str = ""
    negative_history_signal_types: str = ""
    negative_history_time_ms: str = ""


def _row_field(row: object, name: str, default: object) -> object:
    return getattr(row, name, default)


def _serialize_negative_history(
    history: deque[tuple[int, int, int]],
    max_history_len: int,
) -> tuple[str, str, str]:
    if max_history_len <= 0 or len(history) == 0:
        return "", "", ""
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
        return "", "", ""
    video_text = ",".join(str(video_id) for video_id, _, _ in selected)
    signal_text = ",".join(str(signal_type) for _, _, signal_type in selected)
    time_text = ",".join(str(event_time_ms) for _, event_time_ms, _ in selected)
    return video_text, signal_text, time_text

def build_eval_user_samples(
    interactions: pd.DataFrame,
    positive_label_mode: str,
    max_users: Optional[int] = None,
) -> List[EvalUserSample]:
    frame = interactions.copy()
    frame = frame.sort_values(["user_id", "time_ms"], kind="mergesort")
    pos = positive_mask(frame, positive_label_mode)
    frame["is_positive"] = pos.astype(np.int8)

    out: List[EvalUserSample] = []
    for user_id, group in frame.groupby("user_id", sort=False):
        positives = set(group.loc[group["is_positive"] == 1, "video_id"].astype(int).tolist())
        if not positives:
            continue
        last = group.iloc[-1]
        out.append(
            EvalUserSample(
                user_id=int(user_id),
                tab=int(last["tab"]),
                date=int(last.get("date", 0)),
                hourmin=int(last.get("hourmin", 0)),
                hour_bucket=int(last["hour_bucket"]),
                date_bucket=int(last["date_bucket"]),
                time_ms=int(last["time_ms"]),
                history=str(last["history_pos_video_ids"]),
                click_history=str(last.get("history_click_video_ids", last.get("history_pos_video_ids", ""))),
                content_history_video_ids=str(last.get("content_history_video_ids", "")),
                content_history_signal_types=str(last.get("content_history_signal_types", "")),
                content_history_time_ms=str(last.get("content_history_time_ms", "")),
                strong_history_video_ids=str(last.get("strong_history_video_ids", "")),
                strong_history_signal_types=str(last.get("strong_history_signal_types", "")),
                strong_history_time_ms=str(last.get("strong_history_time_ms", "")),
                positives=positives,
            )
        )
        if max_users is not None and len(out) >= max_users:
            break
    return out


def build_split_start_eval_samples(
    interactions: pd.DataFrame,
    positive_label_mode: str,
    max_users: Optional[int] = None,
) -> List[EvalUserSample]:
    """
    按每个用户在当前 split 的起始时间步构造评测样本。

    - 用户特征取该用户在当前 split 中最早时间步的上下文和其之前的正反馈历史。
    - 监督目标取该最早时间步上的正样本 item 集合。
    - 若最早时间步没有正样本，则跳过该用户。
    """
    frame = interactions.copy()
    # Preserve the original row order inside the same timestep. This keeps the
    # first row aligned with the pre-step history on previously exported files.
    frame = frame.sort_values(["user_id", "time_ms"], kind="mergesort")
    pos = positive_mask(frame, positive_label_mode)
    frame["is_positive"] = pos.astype(np.int8)

    out: List[EvalUserSample] = []
    for user_id, group in frame.groupby("user_id", sort=False):
        start_time_ms = int(group.iloc[0]["time_ms"])
        start_step = group[group["time_ms"] == start_time_ms]
        positives = set(start_step.loc[start_step["is_positive"] == 1, "video_id"].astype(int).tolist())
        if not positives:
            continue
        first = start_step.iloc[0]
        out.append(
            EvalUserSample(
                user_id=int(user_id),
                tab=int(first["tab"]),
                date=int(first.get("date", 0)),
                hourmin=int(first.get("hourmin", 0)),
                hour_bucket=int(first["hour_bucket"]),
                date_bucket=int(first["date_bucket"]),
                time_ms=int(first["time_ms"]),
                history=str(first["history_pos_video_ids"]),
                click_history=str(first.get("history_click_video_ids", first.get("history_pos_video_ids", ""))),
                content_history_video_ids=str(first.get("content_history_video_ids", "")),
                content_history_signal_types=str(first.get("content_history_signal_types", "")),
                content_history_time_ms=str(first.get("content_history_time_ms", "")),
                strong_history_video_ids=str(first.get("strong_history_video_ids", "")),
                strong_history_signal_types=str(first.get("strong_history_signal_types", "")),
                strong_history_time_ms=str(first.get("strong_history_time_ms", "")),
                positives=positives,
            )
        )
        if max_users is not None and len(out) >= max_users:
            break
    return out


def build_pointwise_eval_samples(
    interactions: pd.DataFrame,
    positive_label_mode: str,
    max_samples: Optional[int] = None,
    max_negative_history_len: int = 200,
) -> List[EvalUserSample]:
    """
    按曝光点构造评测样本。

    - 每条正样本曝光单独作为一个评测点。
    - 使用该曝光发生前的历史序列，因此同一用户在 valid/test 中的历史会逐步增长。
    - 当前曝光的目标 item 作为单点监督目标。
    """
    frame = interactions.copy()
    frame = frame.sort_values(["user_id", "time_ms"], kind="mergesort").reset_index(drop=True)
    pos = positive_mask(frame, positive_label_mode).to_numpy(dtype=bool, copy=False)
    out: List[EvalUserSample] = []
    negative_histories: Dict[int, deque[tuple[int, int, int]]] = defaultdict(
        lambda: deque(maxlen=max(1, int(max_negative_history_len)))
    )
    frame["__is_positive__"] = pos.astype(np.int8)
    for (_, _), group in frame.groupby(["user_id", "time_ms"], sort=False):
        first = group.iloc[0]
        user_id = int(first["user_id"])
        negative_video_ids, negative_signal_types, negative_time_ms = _serialize_negative_history(
            negative_histories[user_id],
            max_history_len=max_negative_history_len,
        )
        positives = group[group["__is_positive__"] == 1]
        for row in positives.itertuples(index=False):
            out.append(
                EvalUserSample(
                    user_id=user_id,
                    tab=int(row.tab),
                    date=int(_row_field(row, "date", 0)),
                    hourmin=int(_row_field(row, "hourmin", 0)),
                    hour_bucket=int(row.hour_bucket),
                    date_bucket=int(row.date_bucket),
                    time_ms=int(_row_field(row, "time_ms", 0)),
                    history=str(row.history_pos_video_ids),
                    click_history=str(_row_field(row, "history_click_video_ids", _row_field(row, "history_pos_video_ids", ""))),
                    content_history_video_ids=str(_row_field(row, "content_history_video_ids", "")),
                    content_history_signal_types=str(_row_field(row, "content_history_signal_types", "")),
                    content_history_time_ms=str(_row_field(row, "content_history_time_ms", "")),
                    strong_history_video_ids=str(_row_field(row, "strong_history_video_ids", "")),
                    strong_history_signal_types=str(_row_field(row, "strong_history_signal_types", "")),
                    strong_history_time_ms=str(_row_field(row, "strong_history_time_ms", "")),
                    negative_history_video_ids=negative_video_ids,
                    negative_history_signal_types=negative_signal_types,
                    negative_history_time_ms=negative_time_ms,
                    positives={int(row.video_id)},
                )
            )
            if max_samples is not None and len(out) >= max_samples:
                break
        if max_samples is not None and len(out) >= max_samples:
            break

        group_time_ms = int(first.get("time_ms", 0))
        weak_negative_ids = []
        if len(positives) > 0:
            weak_negative_ids = np.unique(
                group.loc[
                    (group["__is_positive__"] == 0) & (group["is_hate"].fillna(0).astype(np.int8) == 0),
                    "video_id",
                ].to_numpy(dtype=np.int64)
            ).tolist()
        strong_negative_ids = np.unique(
            group.loc[group["is_hate"].fillna(0).astype(np.int8) == 1, "video_id"].to_numpy(dtype=np.int64)
        ).tolist()
        for video_id in weak_negative_ids:
            negative_histories[user_id].append((int(video_id), group_time_ms, 1))
        for video_id in strong_negative_ids:
            negative_histories[user_id].append((int(video_id), group_time_ms, 2))
    return out


@dataclass
class ContentRecallAssets:
    item_ids: np.ndarray
    item_vecs: torch.Tensor

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


def load_content_recall_assets(
    embedding_path: Path,
    video_id_to_index_path: Path,
    candidate_item_ids: Optional[np.ndarray],
    device: torch.device,
) -> ContentRecallAssets:
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
        dim = int(emb.shape[1]) if emb.ndim == 2 else 0
        return ContentRecallAssets(
            item_ids=np.zeros((0,), dtype=np.int64),
            item_vecs=torch.zeros((0, dim), dtype=torch.float32, device=device),
        )

    indices = np.asarray([mapping[int(video_id)] for video_id in item_ids.tolist()], dtype=np.int64)
    vecs = np.ascontiguousarray(emb[indices], dtype=np.float32)
    vecs_t = torch.from_numpy(vecs).to(device)
    vecs_t = torch.nn.functional.normalize(vecs_t, p=2, dim=-1)
    return ContentRecallAssets(item_ids=item_ids, item_vecs=vecs_t)


def build_content_user_vectors(
    samples: Sequence[EvalUserSample],
    content_assets: ContentRecallAssets,
    history_len: int,
    strong_weight: float,
    weak_weight: float,
    decay_half_life_hours: float,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    dim = int(content_assets.item_vecs.shape[1]) if content_assets.item_vecs.ndim == 2 else 0
    if len(samples) == 0:
        return torch.zeros((0, dim), dtype=torch.float32, device=device), np.zeros((0,), dtype=bool)
    if len(content_assets.item_ids) == 0 or dim == 0:
        return torch.zeros((len(samples), dim), dtype=torch.float32, device=device), np.zeros((len(samples),), dtype=bool)

    out = []
    valid_rows = np.zeros((len(samples),), dtype=bool)
    half_life_ms = max(float(decay_half_life_hours), 0.0) * 3600.0 * 1000.0

    for sample_idx, sample in enumerate(samples):
        item_ids = parse_history_field(sample.content_history_video_ids, max_history_len=0)
        signal_types = parse_history_field(sample.content_history_signal_types, max_history_len=0)
        event_times = parse_history_field(sample.content_history_time_ms, max_history_len=0)
        if history_len > 0:
            item_ids = item_ids[:history_len]
            signal_types = signal_types[:history_len]
            event_times = event_times[:history_len]
        take = min(len(item_ids), len(signal_types), len(event_times))
        if take <= 0:
            out.append(torch.zeros((dim,), dtype=torch.float32, device=device))
            continue

        item_ids = item_ids[:take]
        signal_types = signal_types[:take]
        event_times = event_times[:take]
        item_idx, valid = content_assets.lookup_indices(item_ids)
        if not np.any(valid):
            out.append(torch.zeros((dim,), dtype=torch.float32, device=device))
            continue

        selected_indices = item_idx[valid]
        selected_signal_types = [signal_types[i] for i, keep in enumerate(valid.tolist()) if keep]
        selected_event_times = [event_times[i] for i, keep in enumerate(valid.tolist()) if keep]
        weights: List[float] = []
        for signal_type, event_time_ms in zip(selected_signal_types, selected_event_times):
            base_weight = strong_weight if int(signal_type) >= 2 else weak_weight
            if half_life_ms > 0.0:
                age_ms = max(0.0, float(sample.time_ms) - float(event_time_ms))
                decay = 0.5 ** (age_ms / half_life_ms)
            else:
                decay = 1.0
            weights.append(float(base_weight) * float(decay))
        weight_sum = float(sum(weights))
        if weight_sum <= 0.0:
            out.append(torch.zeros((dim,), dtype=torch.float32, device=device))
            continue

        emb = content_assets.item_vecs[torch.from_numpy(selected_indices).to(device=device, dtype=torch.long)]
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)
        user_vec = (emb * weight_t).sum(dim=0) / weight_t.sum().clamp_min(1e-6)
        user_vec = torch.nn.functional.normalize(user_vec, p=2, dim=-1)
        out.append(user_vec)
        valid_rows[sample_idx] = True

    return torch.stack(out, dim=0), valid_rows


def merge_recall_candidate_lists(
    main_candidates: Sequence[int],
    content_candidates: Sequence[int],
    max_candidates: Optional[int] = None,
) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for source in (main_candidates, content_candidates):
        for item_id in source:
            item = int(item_id)
            if item < 0 or item in seen:
                continue
            out.append(item)
            seen.add(item)
            if max_candidates is not None and max_candidates > 0 and len(out) >= max_candidates:
                return out
    return out

"""
测评指标函数
"""
def hit_rate_at_k(pred_items: Sequence[int], gt_items: set[int], k: int) -> float:
    if not gt_items:
        return 0.0
    topk = pred_items[:k]
    return 1.0 if any(item in gt_items for item in topk) else 0.0


def recall_at_k(pred_items: Sequence[int], gt_items: set[int], k: int) -> float:
    if not gt_items:
        return 0.0
    topk = pred_items[:k]
    hit = sum(1 for item in topk if item in gt_items)
    return hit / max(1, len(gt_items))


def ndcg_at_k(pred_items: Sequence[int], gt_items: set[int], k: int) -> float:
    if not gt_items:
        return 0.0
    topk = pred_items[:k]
    dcg = 0.0
    for rank, item in enumerate(topk, start=1):
        if item in gt_items:
            dcg += 1.0 / math.log2(rank + 1.0)
    ideal_hits = min(k, len(gt_items))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(2, ideal_hits + 2))
    return dcg / max(idcg, 1e-12)


def mrr_at_k(pred_items: Sequence[int], gt_items: set[int], k: int) -> float:
    topk = pred_items[:k]
    for rank, item in enumerate(topk, start=1):
        if item in gt_items:
            return 1.0 / rank
    return 0.0


def parse_topk(topk: str) -> List[int]:
    out = []
    for token in topk.split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    out = sorted(set(out))
    return out


def _exact_topk_inner_product_search(
    query_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    item_ids: np.ndarray,
    topk: int,
    query_batch_size: int,
    show_progress: bool,
    progress_desc: str,
) -> np.ndarray:
    if query_vecs.ndim == 1:
        query_vecs = query_vecs.unsqueeze(0)
    num_queries = int(query_vecs.shape[0])
    if num_queries == 0 or len(item_ids) == 0 or item_vecs.numel() == 0:
        return np.zeros((num_queries, 0), dtype=np.int64)

    max_k = min(int(topk), int(len(item_ids)))
    if max_k <= 0:
        return np.zeros((num_queries, 0), dtype=np.int64)

    item_ids_t = torch.from_numpy(item_ids.astype(np.int64, copy=False)).to(item_vecs.device)
    out: List[np.ndarray] = []
    batch_starts = range(0, num_queries, query_batch_size)
    iterator = tqdm(
        batch_starts,
        total=(num_queries + query_batch_size - 1) // query_batch_size,
        desc=f"{progress_desc} Exact",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for start in iterator:
            end = min(num_queries, start + query_batch_size)
            scores = torch.matmul(query_vecs[start:end], item_vecs.t())
            idx = torch.topk(scores, k=max_k, dim=1).indices
            pred = item_ids_t[idx].detach().cpu().numpy()
            out.append(pred.astype(np.int64, copy=False))
    return np.concatenate(out, axis=0)


def batched_topk_inner_product_search(
    query_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    item_ids: np.ndarray,
    topk: int,
    query_batch_size: int = 64,
    show_progress: bool = False,
    progress_desc: str = "Top-k Search",
) -> np.ndarray:
    """Exact full-corpus top-k inner-product retrieval."""
    return _exact_topk_inner_product_search(
        query_vecs=query_vecs,
        item_vecs=item_vecs,
        item_ids=item_ids,
        topk=topk,
        query_batch_size=query_batch_size,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_ids_to_npy(path: Path, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ids.astype(np.int64))


def read_ids_from_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    return arr.astype(np.int64)
