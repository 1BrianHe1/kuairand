from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hstu_user_features import HSTUUserFeatureStore


def _parse_sequence_field(raw: str) -> list[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    if text[0] == "[" and text[-1] == "]":
        text = text[1:-1]
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


@dataclass
class KuaiRandHSTUSample:
    user_id: int
    num_targets: int
    seq_item_ids: list[int]
    seq_signals: list[int]
    seq_timestamps: list[int]
    target_item_ids: list[int]


class KuaiRandHSTUSequenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        max_uih_len: int,
        num_targets: int = 1,
        ignore_last_n: int = 0,
        min_total_len: int = 2,
    ) -> None:
        super().__init__()
        if max_uih_len <= 0:
            raise ValueError("max_uih_len must be > 0")
        if num_targets <= 0:
            raise ValueError("num_targets must be > 0")
        if ignore_last_n < 0:
            raise ValueError("ignore_last_n must be >= 0")

        self.csv_path = Path(csv_path)
        self.max_uih_len = int(max_uih_len)
        self.num_targets = int(num_targets)
        self.ignore_last_n = int(ignore_last_n)
        self.min_total_len = int(min_total_len)
        self.samples: list[KuaiRandHSTUSample] = []
        self._load()

    def _load(self) -> None:
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_seq = _parse_sequence_field(row["sequence_item_ids"])
                signal_seq = _parse_sequence_field(row["sequence_ratings"])
                ts_seq = _parse_sequence_field(row["sequence_timestamps"])
                if not (len(item_seq) == len(signal_seq) == len(ts_seq)):
                    continue
                if self.ignore_last_n > 0:
                    if len(item_seq) <= self.ignore_last_n:
                        continue
                    item_seq = item_seq[:-self.ignore_last_n]
                    signal_seq = signal_seq[:-self.ignore_last_n]
                    ts_seq = ts_seq[:-self.ignore_last_n]
                need_len = max(self.min_total_len, self.num_targets + 1)
                if len(item_seq) < need_len:
                    continue

                targets_i = item_seq[-self.num_targets :]
                targets_s = signal_seq[-self.num_targets :]
                targets_t = ts_seq[-self.num_targets :]
                uih_i = item_seq[: -self.num_targets]
                uih_s = signal_seq[: -self.num_targets]
                uih_t = ts_seq[: -self.num_targets]
                if len(uih_i) > self.max_uih_len:
                    uih_i = uih_i[-self.max_uih_len :]
                    uih_s = uih_s[-self.max_uih_len :]
                    uih_t = uih_t[-self.max_uih_len :]
                if len(uih_i) == 0:
                    continue

                self.samples.append(
                    KuaiRandHSTUSample(
                        user_id=int(row["user_id"]),
                        num_targets=len(targets_i),
                        seq_item_ids=uih_i + targets_i,
                        seq_signals=uih_s + targets_s,
                        seq_timestamps=uih_t + targets_t,
                        target_item_ids=targets_i,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> KuaiRandHSTUSample:
        return self.samples[idx]


def kuairand_hstu_collate_fn(
    samples: list[KuaiRandHSTUSample],
    user_feature_store: HSTUUserFeatureStore | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    seq_lengths = torch.tensor([len(s.seq_item_ids) for s in samples], dtype=torch.int64)
    num_targets = torch.tensor([s.num_targets for s in samples], dtype=torch.int64)
    batch_size = len(samples)
    max_len = int(seq_lengths.max().item())
    max_targets = int(num_targets.max().item())

    dense_seq_item_ids = torch.zeros((batch_size, max_len), dtype=torch.int64)
    dense_seq_timestamps = torch.zeros((batch_size, max_len), dtype=torch.int64)
    dense_seq_signals = torch.zeros((batch_size, max_len), dtype=torch.int64)
    target_item_ids = torch.zeros((batch_size, max_targets), dtype=torch.int64)

    for i, sample in enumerate(samples):
        length = len(sample.seq_item_ids)
        dense_seq_item_ids[i, :length] = torch.tensor(sample.seq_item_ids, dtype=torch.int64)
        dense_seq_timestamps[i, :length] = torch.tensor(sample.seq_timestamps, dtype=torch.int64)
        dense_seq_signals[i, :length] = torch.tensor(sample.seq_signals, dtype=torch.int64)
        nt = len(sample.target_item_ids)
        target_item_ids[i, :nt] = torch.tensor(sample.target_item_ids, dtype=torch.int64)

    user_ids = torch.tensor([s.user_id for s in samples], dtype=torch.int64)
    batch = {
        "user_ids": user_ids,
        "seq_lengths": seq_lengths,
        "num_targets": num_targets,
        "dense_seq_item_ids": dense_seq_item_ids,
        "dense_seq_timestamps": dense_seq_timestamps,
        "dense_seq_signals": dense_seq_signals,
        "target_item_ids": target_item_ids,
    }
    if user_feature_store is not None:
        user_ids_np = np.asarray(user_ids.tolist(), dtype=np.int64)
        user_static_cat, user_static_num = user_feature_store.lookup(user_ids_np)
        batch["user_static_cat"] = torch.from_numpy(user_static_cat)
        batch["user_static_num"] = torch.from_numpy(user_static_num)
    return batch


def create_kuairand_hstu_dataloader(
    csv_path: str | Path,
    batch_size: int,
    max_uih_len: int,
    num_targets: int = 1,
    ignore_last_n: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    user_feature_store: HSTUUserFeatureStore | None = None,
) -> DataLoader:
    dataset = KuaiRandHSTUSequenceDataset(
        csv_path=csv_path,
        max_uih_len=max_uih_len,
        num_targets=num_targets,
        ignore_last_n=ignore_last_n,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(
            kuairand_hstu_collate_fn,
            user_feature_store=user_feature_store,
        ),
        drop_last=False,
    )
