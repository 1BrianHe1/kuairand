from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from data_utils import (
    DEFAULT_BUCKET_SIZES,
    _normalize_with_stats,
    _safe_numeric,
    bucketize_int_array,
    bucketize_str_series,
)


HSTU_USER_CATEGORICAL_COLUMNS = [
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
]

HSTU_USER_NUMERIC_COLUMNS = [
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
]


def resolve_hstu_user_features_csv(
    user_features_csv: Path | None,
    data_dir: Path,
) -> Path:
    if user_features_csv is not None:
        return user_features_csv

    parent_candidate = data_dir.parent / "user_features.selected.csv"
    if parent_candidate.exists():
        return parent_candidate

    local_candidate = data_dir / "user_features.selected.csv"
    if local_candidate.exists():
        return local_candidate

    raise FileNotFoundError(
        "Could not resolve HSTU user static feature csv. "
        "Pass --user-features-csv explicitly or place user_features.selected.csv "
        f"under {data_dir.parent}."
    )


@dataclass
class HSTUUserFeatureStore:
    user_ids: np.ndarray
    cat_features: np.ndarray
    num_features: np.ndarray
    bucket_sizes: Dict[str, int]
    num_mean: np.ndarray
    num_std: np.ndarray

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        bucket_sizes: Dict[str, int] | None = None,
    ) -> "HSTUUserFeatureStore":
        sizes = dict(DEFAULT_BUCKET_SIZES)
        if bucket_sizes is not None:
            sizes.update(bucket_sizes)

        cols = ["user_id"] + HSTU_USER_CATEGORICAL_COLUMNS + HSTU_USER_NUMERIC_COLUMNS
        df = pd.read_csv(csv_path, usecols=cols).fillna(0)

        cat_parts = []
        for col in HSTU_USER_CATEGORICAL_COLUMNS:
            bucket = int(sizes[col])
            if col == "user_active_degree":
                cat = bucketize_str_series(df[col], bucket, prefix=col)
            else:
                values = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
                cat = bucketize_int_array(values, bucket)
            cat_parts.append(cat[:, None])
        cat_arr = (
            np.concatenate(cat_parts, axis=1).astype(np.int64, copy=False)
            if cat_parts
            else np.zeros((len(df), 0), dtype=np.int64)
        )

        num_raw = df[HSTU_USER_NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
        num_log = _safe_numeric(num_raw)
        mean = num_log.mean(axis=0, keepdims=True)
        std = num_log.std(axis=0, keepdims=True)
        num_arr = _normalize_with_stats(num_log, mean, std).astype(np.float32, copy=False)

        user_ids = pd.to_numeric(df["user_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        order = np.argsort(user_ids, kind="mergesort")
        return cls(
            user_ids=user_ids[order],
            cat_features=cat_arr[order],
            num_features=num_arr[order],
            bucket_sizes={col: int(sizes[col]) for col in HSTU_USER_CATEGORICAL_COLUMNS},
            num_mean=mean.reshape(-1).astype(np.float32, copy=False),
            num_std=std.reshape(-1).astype(np.float32, copy=False),
        )

    def lookup(self, user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        cat = np.zeros((len(user_ids), self.cat_features.shape[1]), dtype=np.int64)
        num = np.zeros((len(user_ids), self.num_features.shape[1]), dtype=np.float32)
        if len(self.user_ids) == 0 or len(user_ids) == 0:
            return cat, num

        idx = np.searchsorted(self.user_ids, user_ids)
        idx_clip = np.clip(idx, 0, len(self.user_ids) - 1)
        valid = (idx < len(self.user_ids)) & (self.user_ids[idx_clip] == user_ids)
        if np.any(valid):
            cat[valid] = self.cat_features[idx[valid]]
            num[valid] = self.num_features[idx[valid]]
        return cat, num

    @property
    def num_numeric_features(self) -> int:
        return int(self.num_features.shape[1])
