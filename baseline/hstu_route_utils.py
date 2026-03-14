from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import EvalUserSample, batched_topk_inner_product_search
from hstu_kuairand_model import KuaiRandHSTURecModel
from hstu_user_features import HSTUUserFeatureStore, resolve_hstu_user_features_csv


@dataclass
class HSTURouteEvalRow:
    user_id: int
    time_ms: int
    target_video_id: int
    seq_item_ids: List[int]
    seq_signals: List[int]
    seq_timestamps: List[int]
    target_item_id: int


@dataclass
class HSTURouteAssets:
    model: KuaiRandHSTURecModel
    candidate_video_ids: np.ndarray
    candidate_hstu_item_ids: np.ndarray
    candidate_item_vecs: torch.Tensor
    eval_lookup: Dict[Tuple[int, int, int], HSTURouteEvalRow]
    user_feature_store: HSTUUserFeatureStore | None
    l2_norm_embeddings: bool
    max_uih_len: int


def _parse_sequence_field(raw: str) -> List[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text == "nan":
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _load_hstu_model(ckpt_path: Path, device: torch.device) -> tuple[KuaiRandHSTURecModel, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    metadata = ckpt["metadata"]
    stats = metadata["stats"]
    user_static_bucket_sizes = ckpt.get("user_static_bucket_sizes", {})
    user_static_num_features = int(ckpt.get("user_static_num_features", 0))
    model = KuaiRandHSTURecModel(
        num_items=int(stats["num_items"]),
        num_user_embeddings=int(stats["max_user_id"]) + 1,
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        item_embedding_dim=int(cfg.get("item_embedding_dim", 128)),
        signal_embedding_dim=int(cfg.get("signal_embedding_dim", 16)),
        user_embedding_dim=int(cfg.get("user_embedding_dim", 16)),
        max_sequence_len=int(cfg.get("max_sequence_len", 202)),
        token_layout=str(cfg.get("token_layout", "interleaved")),
        user_feature_mode=str(cfg.get("user_feature_mode", "first_token")),
        num_layers=int(cfg.get("num_layers", 2)),
        num_heads=int(cfg.get("num_heads", 2)),
        linear_dim=int(cfg.get("linear_dim", 64)),
        attention_dim=int(cfg.get("attention_dim", 32)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        attention_activation=str(cfg.get("attention_activation", "silu")),
        use_position_bias=bool(cfg.get("use_position_bias", False)),
        use_time_bias=bool(cfg.get("use_time_bias", False)),
        time_num_buckets=int(cfg.get("time_num_buckets", 128)),
        time_log_base=float(cfg.get("time_log_base", 0.301)),
        scale_by_sqrt_d=bool(cfg.get("scale_by_sqrt_d", False)),
        user_static_bucket_sizes=user_static_bucket_sizes,
        num_user_static_numeric=user_static_num_features,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, metadata


def _load_eval_lookup(
    csv_path: Path,
    hstu_item_to_video: Dict[int, int],
) -> Dict[Tuple[int, int, int], HSTURouteEvalRow]:
    out: Dict[Tuple[int, int, int], HSTURouteEvalRow] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_item_ids = _parse_sequence_field(row.get("sequence_item_ids", ""))
            seq_signals = _parse_sequence_field(row.get("sequence_ratings", ""))
            seq_timestamps = _parse_sequence_field(row.get("sequence_timestamps", ""))
            if not seq_item_ids or len(seq_item_ids) != len(seq_signals) or len(seq_item_ids) != len(seq_timestamps):
                continue
            target_item_id = int(seq_item_ids[-1])
            target_video_id = hstu_item_to_video.get(target_item_id)
            if target_video_id is None:
                continue
            time_ms = int(seq_timestamps[-1])
            user_id = int(row["user_id"])
            key = (user_id, time_ms, target_video_id)
            if key in out:
                continue
            out[key] = HSTURouteEvalRow(
                user_id=user_id,
                time_ms=time_ms,
                target_video_id=target_video_id,
                seq_item_ids=seq_item_ids,
                seq_signals=seq_signals,
                seq_timestamps=seq_timestamps,
                target_item_id=target_item_id,
            )
    return out


def load_hstu_route_assets(
    ckpt_path: Path,
    data_dir: Path,
    candidate_video_ids: np.ndarray,
    split_name: str,
    device: torch.device,
) -> HSTURouteAssets:
    model, cfg, _ = _load_hstu_model(ckpt_path=ckpt_path, device=device)
    user_feature_store = None
    if model.user_static_bucket_sizes or model.num_user_static_numeric > 0:
        user_features_csv = resolve_hstu_user_features_csv(None, data_dir)
        user_feature_store = HSTUUserFeatureStore.from_csv(user_features_csv)
    with (data_dir / "video_id_to_hstu_item_id.json").open("r", encoding="utf-8") as f:
        video_to_hstu = {int(k): int(v) for k, v in json.load(f).items()}
    with (data_dir / "hstu_item_id_to_video_id.json").open("r", encoding="utf-8") as f:
        hstu_to_video = {int(k): int(v) for k, v in json.load(f).items()}

    candidate_video_ids = np.unique(np.asarray(candidate_video_ids, dtype=np.int64))
    filtered_video_ids: List[int] = []
    filtered_hstu_item_ids: List[int] = []
    for video_id in candidate_video_ids.tolist():
        hstu_item_id = video_to_hstu.get(int(video_id))
        if hstu_item_id is None:
            continue
        filtered_video_ids.append(int(video_id))
        filtered_hstu_item_ids.append(int(hstu_item_id))

    candidate_video_arr = np.asarray(filtered_video_ids, dtype=np.int64)
    candidate_hstu_arr = np.asarray(filtered_hstu_item_ids, dtype=np.int64)
    with torch.no_grad():
        if len(candidate_hstu_arr) > 0:
            item_repr = model.get_item_representations(
                torch.from_numpy(candidate_hstu_arr).to(device=device, dtype=torch.long)
            ).detach()
            if bool(cfg.get("l2_norm_embeddings", False)):
                item_repr = F.normalize(item_repr, dim=-1)
        else:
            item_repr = torch.zeros((0, model.embedding_dim), dtype=torch.float32, device=device)

    split_csv = data_dir / f"{split_name}.csv"
    eval_lookup = _load_eval_lookup(split_csv, hstu_item_to_video=hstu_to_video)
    return HSTURouteAssets(
        model=model,
        candidate_video_ids=candidate_video_arr,
        candidate_hstu_item_ids=candidate_hstu_arr,
        candidate_item_vecs=item_repr,
        eval_lookup=eval_lookup,
        user_feature_store=user_feature_store,
        l2_norm_embeddings=bool(cfg.get("l2_norm_embeddings", False)),
        max_uih_len=int(cfg.get("max_uih_len", 100)),
    )


def _truncate_eval_row(row: HSTURouteEvalRow, max_uih_len: int) -> HSTURouteEvalRow:
    if max_uih_len <= 0 or len(row.seq_item_ids) <= max_uih_len + 1:
        return row
    keep_history = max_uih_len
    hist_start = max(0, len(row.seq_item_ids) - 1 - keep_history)
    seq_item_ids = row.seq_item_ids[hist_start:]
    seq_signals = row.seq_signals[hist_start:]
    seq_timestamps = row.seq_timestamps[hist_start:]
    return HSTURouteEvalRow(
        user_id=row.user_id,
        time_ms=row.time_ms,
        target_video_id=row.target_video_id,
        seq_item_ids=seq_item_ids,
        seq_signals=seq_signals,
        seq_timestamps=seq_timestamps,
        target_item_id=row.target_item_id,
    )


def _collate_hstu_rows(
    rows: Sequence[HSTURouteEvalRow],
    user_feature_store: HSTUUserFeatureStore | None = None,
) -> dict:
    batch_size = len(rows)
    seq_lengths = torch.tensor([len(row.seq_item_ids) for row in rows], dtype=torch.int64)
    num_targets = torch.ones((batch_size,), dtype=torch.int64)
    max_len = int(seq_lengths.max().item()) if batch_size > 0 else 0

    dense_seq_item_ids = torch.zeros((batch_size, max_len), dtype=torch.int64)
    dense_seq_timestamps = torch.zeros((batch_size, max_len), dtype=torch.int64)
    dense_seq_signals = torch.zeros((batch_size, max_len), dtype=torch.int64)
    target_item_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
    user_ids = torch.zeros((batch_size,), dtype=torch.int64)

    for i, row in enumerate(rows):
        length = len(row.seq_item_ids)
        dense_seq_item_ids[i, :length] = torch.tensor(row.seq_item_ids, dtype=torch.int64)
        dense_seq_timestamps[i, :length] = torch.tensor(row.seq_timestamps, dtype=torch.int64)
        dense_seq_signals[i, :length] = torch.tensor(row.seq_signals, dtype=torch.int64)
        target_item_ids[i, 0] = int(row.target_item_id)
        user_ids[i] = int(row.user_id)
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
        user_static_cat, user_static_num = user_feature_store.lookup(
            np.asarray(user_ids.tolist(), dtype=np.int64)
        )
        batch["user_static_cat"] = torch.from_numpy(user_static_cat)
        batch["user_static_num"] = torch.from_numpy(user_static_num)
    return batch


def build_hstu_query_vectors(
    samples: Sequence[EvalUserSample],
    assets: HSTURouteAssets,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    dim = int(assets.candidate_item_vecs.shape[1]) if assets.candidate_item_vecs.ndim == 2 else int(assets.model.embedding_dim)
    out = torch.zeros((len(samples), dim), dtype=torch.float32, device=device)
    valid = np.zeros((len(samples),), dtype=bool)

    matched_rows: List[HSTURouteEvalRow] = []
    matched_indices: List[int] = []
    for sample_idx, sample in enumerate(samples):
        if len(sample.positives) != 1:
            continue
        target_video_id = int(next(iter(sample.positives)))
        key = (int(sample.user_id), int(sample.time_ms), target_video_id)
        row = assets.eval_lookup.get(key)
        if row is None:
            continue
        matched_rows.append(_truncate_eval_row(row, max_uih_len=assets.max_uih_len))
        matched_indices.append(sample_idx)

    if not matched_rows:
        return out, valid

    assets.model.eval()
    with torch.no_grad():
        for start in range(0, len(matched_rows), max(1, int(batch_size))):
            end = min(len(matched_rows), start + max(1, int(batch_size)))
            batch_rows = matched_rows[start:end]
            batch_idx = matched_indices[start:end]
            batch = _collate_hstu_rows(
                batch_rows,
                user_feature_store=assets.user_feature_store,
            )
            query_out = assets.model.get_query_output(batch=batch, device=device, causal=True)
            query_states = query_out.query_states
            if assets.l2_norm_embeddings:
                query_states = F.normalize(query_states, dim=-1)
            for local_idx, global_idx in enumerate(batch_idx):
                if bool(query_out.valid_mask[local_idx].item()):
                    out[global_idx] = query_states[local_idx]
                    valid[global_idx] = True
    return out, valid


def build_hstu_candidate_lists(
    samples: Sequence[EvalUserSample],
    assets: HSTURouteAssets | None,
    topn: int,
    device: torch.device,
    query_batch_size: int,
    show_progress: bool,
    progress_desc: str,
) -> tuple[List[List[int]], int]:
    pred_lists: List[List[int]] = [[] for _ in samples]
    if assets is None or topn <= 0 or len(assets.candidate_video_ids) == 0:
        return pred_lists, 0

    query_vecs, valid = build_hstu_query_vectors(
        samples=samples,
        assets=assets,
        device=device,
        batch_size=query_batch_size,
    )
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return pred_lists, 0

    preds = batched_topk_inner_product_search(
        query_vecs=query_vecs[torch.from_numpy(valid_idx).to(device=device, dtype=torch.long)],
        item_vecs=assets.candidate_item_vecs,
        item_ids=assets.candidate_video_ids,
        topk=topn,
        query_batch_size=query_batch_size,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    for pos, sample_idx in enumerate(valid_idx.tolist()):
        pred_lists[sample_idx] = preds[pos].tolist()
    return pred_lists, int(len(valid_idx))
