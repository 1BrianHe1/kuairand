#!/usr/bin/env python3
"""End-to-end evaluation for two-tower recall + single-task DIN+DCN ranker."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_utils import (
    DEFAULT_BUCKET_SIZES,
    DEFAULT_RANK_BUCKET_SIZES,
    ItemFeatureStore,
    RankBatchCollator,
    RankItemFeatureStore,
    RankTrainDataset,
    RankUserFeatureStore,
    UserFeatureStore,
    batched_topk_inner_product_search,
    build_content_user_vectors,
    build_pointwise_eval_samples,
    encode_context_features,
    encode_history_batch,
    encode_rank_context_features,
    hit_rate_at_k,
    load_interactions,
    load_content_recall_assets,
    ndcg_at_k,
    parse_topk,
    read_ids_from_npy,
    save_json,
    set_seed,
)
from hstu_route_utils import build_hstu_candidate_lists, load_hstu_route_assets
from models import ContentTwoTowerRecallModel, DINDCNRanker, TwoTowerRecallModel
from train_recall_content_twotower import (
    _build_candidate_item_bank as _build_content_candidate_item_bank,
    _build_eval_query_features,
    load_category_feature_store,
    load_content_embedding_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KuaiRand two-stage baseline: recall + rank."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure"),
    )
    parser.add_argument(
        "--recall-ckpt",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/recall_pure/recall_model.pt"),
    )
    parser.add_argument(
        "--rank-ckpt",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/rank_pure/rank_model.pt"),
    )
    parser.add_argument(
        "--candidate-item-ids",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/recall_pure/candidate_item_ids.npy"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/eval_pure/test_metrics.json"),
    )
    parser.add_argument("--topk", type=str, default="20,50")
    parser.add_argument("--test-max-rows", type=int, default=None)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument(
        "--recall-topn",
        type=int,
        default=300,
        help="Number of recall candidates kept before reranking.",
    )
    parser.add_argument(
        "--content-recall-topn",
        type=int,
        default=0,
        help="Additional content-recall candidates merged before reranking.",
    )
    parser.add_argument("--content-item-emb", type=Path, default=None)
    parser.add_argument("--content-video-id-to-index", type=Path, default=None)
    parser.add_argument(
        "--content-ckpt",
        type=Path,
        default=Path(
            "/home/hfx/KuaiRand/baseline/checkpoints/"
            "content_twotower_recall_pure_v2/"
            "content_recall_model.pt"
        ),
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
    parser.add_argument("--content-history-len", type=int, default=5)
    parser.add_argument("--content-strong-weight", type=float, default=1.0)
    parser.add_argument("--content-weak-weight", type=float, default=0.5)
    parser.add_argument("--content-decay-half-life-hours", type=float, default=48.0)
    parser.add_argument("--hstu-recall-topn", type=int, default=0)
    parser.add_argument("--hstu-ckpt", type=Path, default=None)
    parser.add_argument("--hstu-data-dir", type=Path, default=None)
    parser.add_argument(
        "--fusion-method",
        choices=["weighted_rrf"],
        default="weighted_rrf",
        help="How to combine main/content/HSTU routes before reranking.",
    )
    parser.add_argument(
        "--route-source-topn",
        type=int,
        default=200,
        help="Per-route retrieval depth before applying fusion budgets.",
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--item-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--rank-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--positive-label-mode",
        choices=["click", "long_view", "click_or_long", "signal_positive"],
        default="click_or_long",
    )
    parser.add_argument("--score-click-weight", type=float, default=1.0)
    parser.add_argument("--max-history-len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--skip-no-candidate-positive",
        action="store_true",
        help="Skip users with no positives in candidate pool when computing recall/e2e metrics.",
    )
    return parser.parse_args()


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _load_recall_model(ckpt_path: Path, device: torch.device) -> tuple[TwoTowerRecallModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    bucket_sizes = ckpt.get("bucket_sizes", dict(DEFAULT_BUCKET_SIZES))
    model = TwoTowerRecallModel(
        bucket_sizes=bucket_sizes,
        embedding_dim=int(cfg.get("embedding_dim", 16)),
        tower_dim=int(cfg.get("tower_dim", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, bucket_sizes


def _load_rank_model(ckpt_path: Path, device: torch.device) -> tuple[DINDCNRanker, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    bucket_sizes = ckpt.get("bucket_sizes", dict(DEFAULT_RANK_BUCKET_SIZES))
    model = DINDCNRanker(
        bucket_sizes=bucket_sizes,
        embedding_dim=int(cfg.get("embedding_dim", 16)),
        hidden_dim=int(cfg.get("hidden_dim", cfg.get("shared_dim", 128))),
        dcn_layers=int(cfg.get("dcn_layers", 3)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, bucket_sizes


def _encode_rank_history_features(
    raw_histories,
    item_store: RankItemFeatureStore,
    bucket_sizes: Dict[str, int],
    max_history_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist_ids, hist_mask, hist_raw = encode_history_batch(
        raw_histories=raw_histories,
        max_history_len=max_history_len,
        item_bucket_size=bucket_sizes["video_id"],
        return_raw_ids=True,
    )
    hist_author_ids = np.zeros_like(hist_ids)
    hist_tag_ids = np.zeros_like(hist_ids)
    if hist_raw.size > 0:
        flat_hist = hist_raw.reshape(-1)
        flat_cat, _, flat_valid = item_store.lookup(flat_hist)
        flat_author = np.zeros_like(flat_hist, dtype=np.int64)
        flat_tag = np.zeros_like(flat_hist, dtype=np.int64)
        if np.any(flat_valid):
            flat_author[flat_valid] = flat_cat[flat_valid, 1]
            flat_tag[flat_valid] = flat_cat[flat_valid, 7]
        hist_author_ids = flat_author.reshape(hist_raw.shape)
        hist_tag_ids = flat_tag.reshape(hist_raw.shape)
    return hist_ids, hist_author_ids, hist_tag_ids, hist_mask


def _load_content_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[ContentTwoTowerRecallModel, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    light_context_bucket_sizes = ckpt.get("light_context_bucket_sizes", {})
    model = ContentTwoTowerRecallModel(
        bucket_sizes=light_context_bucket_sizes,
        content_dim=int(ckpt.get("content_dim", 768)),
        embedding_dim=int(cfg.get("embedding_dim", 16)),
        tower_dim=int(cfg.get("tower_dim", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        long_term_input_dim=int(ckpt.get("long_term_input_dim", 0)),
        negative_immunity_mode=str(cfg.get("negative_immunity_mode", "off")),
        negative_fixed_alpha=float(cfg.get("negative_fixed_alpha", 0.5)),
        negative_max_alpha=float(cfg.get("negative_max_alpha", 1.0)),
        negative_gate_hidden_dim=int(cfg.get("negative_gate_hidden_dim", 64)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, light_context_bucket_sizes


def _build_candidate_ids(
    candidate_item_ids_path: Path,
    fallback_item_path: Path,
) -> np.ndarray:
    if candidate_item_ids_path.exists():
        candidate_ids = read_ids_from_npy(candidate_item_ids_path)
    else:
        item_ids = pd.read_csv(fallback_item_path, usecols=["video_id"])["video_id"].to_numpy(dtype=np.int64)
        candidate_ids = np.unique(item_ids)

    return np.unique(candidate_ids.astype(np.int64))


def _encode_candidate_item_vectors(
    model: TwoTowerRecallModel,
    item_store: ItemFeatureStore,
    candidate_ids: np.ndarray,
    device: torch.device,
    item_batch_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    cat, num, valid = item_store.lookup(candidate_ids)
    ids = candidate_ids[valid]
    cat = cat[valid]
    num = num[valid]

    vectors = []
    with torch.no_grad():
        starts = range(0, len(ids), item_batch_size)
        iterator = tqdm(
            starts,
            total=(len(ids) + item_batch_size - 1) // item_batch_size if len(ids) > 0 else 0,
            desc="Eval Candidate Encode",
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for start in iterator:
            end = min(len(ids), start + item_batch_size)
            item_cat_t = torch.from_numpy(cat[start:end]).to(device)
            item_num_t = torch.from_numpy(num[start:end]).to(device)
            v = model.encode_item(item_cat=item_cat_t, item_num=item_num_t).detach()
            vectors.append(v)
    if vectors:
        all_vecs = torch.cat(vectors, dim=0)
    else:
        all_vecs = torch.zeros((0, 1), dtype=torch.float32, device=device)
    return ids, all_vecs


def _build_learned_content_candidate_lists(
    eval_samples,
    content_model: ContentTwoTowerRecallModel,
    content_cfg: dict,
    content_store,
    category_store,
    device: torch.device,
    candidate_topn: int,
    item_batch_size: int,
    query_batch_size: int,
    show_progress: bool,
) -> List[List[int]]:
    if len(eval_samples) == 0 or candidate_topn <= 0:
        return []
    cand_item_ids, cand_item_vecs = _build_content_candidate_item_bank(
        model=content_model,
        content_store=content_store,
        device=device,
        batch_size=item_batch_size,
        show_progress=show_progress,
    )
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
        samples=eval_samples,
        content_store=content_store,
        category_store=category_store,
        short_history_len=int(content_cfg.get("short_history_len", 15)),
        long_term_history_len=int(content_cfg.get("long_term_history_len", 100)),
        negative_history_len=int(content_cfg.get("negative_history_len", 10)),
        short_signal_weight_map={
            1: float(content_cfg.get("click_weight", 0.5)),
            2: float(content_cfg.get("long_view_weight", 1.0)),
            3: float(content_cfg.get("like_weight", 1.25)),
            4: float(content_cfg.get("social_weight", 1.5)),
        },
        long_signal_weight_map={
            2: float(content_cfg.get("long_view_weight", 1.0)),
            3: float(content_cfg.get("like_weight", 1.25)),
            4: float(content_cfg.get("social_weight", 1.5)),
        },
        negative_signal_weight_map={
            1: float(content_cfg.get("negative_relative_weight", 0.35)),
            2: float(content_cfg.get("negative_hate_weight", 1.0)),
        },
        negative_semantic_similarity_threshold=float(
            content_cfg.get("negative_semantic_sim_threshold", 0.1)
        ),
        short_half_life_ms=max(float(content_cfg.get("short_history_half_life_hours", 48.0)), 0.0)
        * 3600.0
        * 1000.0,
        negative_half_life_ms=max(float(content_cfg.get("negative_half_life_hours", 72.0)), 0.0)
        * 3600.0
        * 1000.0,
    )
    with torch.no_grad():
        user_vecs = content_model.encode_user(
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
            topk=candidate_topn,
            query_batch_size=query_batch_size,
            show_progress=show_progress,
            progress_desc="Eval Learned Content Search",
        )
    return [row.tolist() for row in top_items]


def _fuse_weighted_rrf(
    route_lists: Dict[str, List[int]],
    route_weights: Dict[str, float],
    total_budget: int,
    rrf_k: float,
) -> List[int]:
    score_map: Dict[int, float] = {}
    best_rank_map: Dict[int, int] = {}
    route_order = {"hstu": 0, "main": 1, "content": 2}
    route_best_source: Dict[int, str] = {}

    for route_name, items in route_lists.items():
        route_weight = float(route_weights.get(route_name, 0.0))
        if route_weight <= 0.0:
            continue
        for rank_idx, item_id in enumerate(items, start=1):
            value = int(item_id)
            score = route_weight / (float(rrf_k) + float(rank_idx))
            score_map[value] = score_map.get(value, 0.0) + score
            current_best_rank = best_rank_map.get(value, 10**9)
            if rank_idx < current_best_rank:
                best_rank_map[value] = rank_idx
                route_best_source[value] = route_name

    ranked = sorted(
        score_map.items(),
        key=lambda kv: (
            -kv[1],
            best_rank_map.get(kv[0], 10**9),
            route_order.get(route_best_source.get(kv[0], "content"), 99),
            kv[0],
        ),
    )
    return [item_id for item_id, _ in ranked[:total_budget]]


def _build_fused_candidates(
    *,
    fusion_method: str,
    main_candidates: List[int],
    content_candidates: List[int],
    hstu_candidates: List[int],
    main_budget: int,
    content_budget: int,
    hstu_budget: int,
    rrf_k: float,
) -> List[int]:
    trimmed_main = main_candidates[: max(0, main_budget)]
    trimmed_content = content_candidates[: max(0, content_budget)]
    trimmed_hstu = hstu_candidates[: max(0, hstu_budget)]
    total_budget = max(0, main_budget) + max(0, content_budget) + max(0, hstu_budget)

    if total_budget <= 0:
        return []
    route_weights = {
        "main": float(max(0, main_budget)) / float(total_budget),
        "content": float(max(0, content_budget)) / float(total_budget),
        "hstu": float(max(0, hstu_budget)) / float(total_budget),
    }
    return _fuse_weighted_rrf(
        {
            "hstu": trimmed_hstu,
            "main": trimmed_main,
            "content": trimmed_content,
        },
        route_weights=route_weights,
        total_budget=total_budget,
        rrf_k=rrf_k,
    )


def evaluate_recall_and_rerank(
    recall_model: TwoTowerRecallModel,
    rank_model: DINDCNRanker,
    eval_samples,
    recall_user_store: UserFeatureStore,
    recall_item_store: ItemFeatureStore,
    rank_user_store: RankUserFeatureStore,
    rank_item_store: RankItemFeatureStore,
    candidate_item_ids: np.ndarray,
    candidate_item_vecs: torch.Tensor,
    recall_bucket_sizes: Dict[str, int],
    rank_bucket_sizes: Dict[str, int],
    topk: List[int],
    recall_topn: int,
    max_history_len: int,
    score_click_weight: float,
    device: torch.device,
    query_batch_size: int,
    skip_no_candidate_positive: bool,
    show_progress: bool,
    content_assets=None,
    content_route_mode: str = "zero_shot",
    learned_content_assets=None,
    content_recall_topn: int = 0,
    content_history_len: int = 5,
    content_strong_weight: float = 1.0,
    content_weak_weight: float = 0.5,
    content_decay_half_life_hours: float = 48.0,
    hstu_assets=None,
    hstu_recall_topn: int = 0,
    fusion_method: str = "weighted_rrf",
    route_source_topn: int = 200,
    rrf_k: float = 60.0,
) -> Dict[str, float]:
    if len(eval_samples) == 0 or len(candidate_item_ids) == 0:
        out = {f"recall_hr@{k}": 0.0 for k in topk}
        out.update({f"recall_ndcg@{k}": 0.0 for k in topk})
        out.update({f"e2e_hr@{k}": 0.0 for k in topk})
        out.update({f"e2e_ndcg@{k}": 0.0 for k in topk})
        out.update({f"route_main_hr@{k}": 0.0 for k in topk})
        out.update({f"route_main_ndcg@{k}": 0.0 for k in topk})
        out.update({f"route_content_hr@{k}": 0.0 for k in topk})
        out.update({f"route_content_ndcg@{k}": 0.0 for k in topk})
        out.update({f"route_hstu_hr@{k}": 0.0 for k in topk})
        out.update({f"route_hstu_ndcg@{k}": 0.0 for k in topk})
        out["num_users"] = 0.0
        out["num_eval_points"] = 0.0
        out["num_unique_users"] = 0.0
        out["num_skipped"] = 0.0
        out["avg_recall_candidates"] = 0.0
        out["avg_main_candidates"] = 0.0
        out["avg_content_candidates"] = 0.0
        out["avg_hstu_candidates"] = 0.0
        out["num_hstu_queries"] = 0.0
        return out

    route_source_topn = max(int(route_source_topn), max(topk), 1)
    main_budget = max(int(recall_topn), 0)
    content_budget = max(int(content_recall_topn), 0)
    hstu_budget = max(int(hstu_recall_topn), 0)
    cand_set = set(candidate_item_ids.tolist())

    totals = {f"recall_hr@{k}": 0.0 for k in topk}
    totals.update({f"recall_ndcg@{k}": 0.0 for k in topk})
    totals.update({f"e2e_hr@{k}": 0.0 for k in topk})
    totals.update({f"e2e_ndcg@{k}": 0.0 for k in topk})
    route_main_totals = {f"route_main_hr@{k}": 0.0 for k in topk}
    route_main_totals.update({f"route_main_ndcg@{k}": 0.0 for k in topk})
    route_content_totals = {f"route_content_hr@{k}": 0.0 for k in topk}
    route_content_totals.update({f"route_content_ndcg@{k}": 0.0 for k in topk})
    route_hstu_totals = {f"route_hstu_hr@{k}": 0.0 for k in topk}
    route_hstu_totals.update({f"route_hstu_ndcg@{k}": 0.0 for k in topk})
    skipped = 0
    recall_candidate_total = 0.0
    main_candidate_total = 0.0
    content_candidate_total = 0.0
    hstu_candidate_total = 0.0
    active_samples = []
    user_ids = []
    tab = []
    date = []
    hourmin = []
    hour = []
    histories = []
    click_histories = []
    for sample in eval_samples:
        gt = sample.positives
        if skip_no_candidate_positive and len(gt.intersection(cand_set)) == 0:
            skipped += 1
            continue
        active_samples.append(sample)
        user_ids.append(sample.user_id)
        tab.append(sample.tab)
        date.append(sample.date)
        hourmin.append(sample.hourmin)
        hour.append(sample.hour_bucket)
        histories.append(sample.history)
        click_histories.append(sample.click_history)

    if not active_samples:
        out = {k: 0.0 for k in totals}
        out["num_users"] = 0.0
        out["num_eval_points"] = 0.0
        out["num_unique_users"] = 0.0
        out["num_skipped"] = float(skipped)
        out["num_candidates"] = float(len(candidate_item_ids))
        out["avg_recall_candidates"] = 0.0
        out["avg_main_candidates"] = 0.0
        out["avg_content_candidates"] = 0.0
        out["avg_hstu_candidates"] = 0.0
        out["num_hstu_queries"] = 0.0
        for k in topk:
            out[f"route_main_hr@{k}"] = 0.0
            out[f"route_main_ndcg@{k}"] = 0.0
            out[f"route_content_hr@{k}"] = 0.0
            out[f"route_content_ndcg@{k}"] = 0.0
            out[f"route_hstu_hr@{k}"] = 0.0
            out[f"route_hstu_ndcg@{k}"] = 0.0
        return out

    user_ids_arr = np.asarray(user_ids, dtype=np.int64)
    tab_arr = np.asarray(tab, dtype=np.int64)
    date_arr = np.asarray(date, dtype=np.int64)
    hourmin_arr = np.asarray(hourmin, dtype=np.int64)
    hour_arr = np.asarray(hour, dtype=np.int64)
    date_bucket_arr = np.asarray([sample.date_bucket for sample in active_samples], dtype=np.int64)
    time_ms_arr = np.asarray([sample.time_ms for sample in active_samples], dtype=np.int64)
    recall_user_cat, recall_user_num = recall_user_store.lookup(user_ids_arr)
    recall_ctx_cat = encode_context_features(tab_arr, hour_arr, date_bucket_arr, recall_bucket_sizes)
    hist_ids, hist_mask = encode_history_batch(
        raw_histories=histories,
        max_history_len=max_history_len,
        item_bucket_size=recall_bucket_sizes["video_id"],
    )
    rank_user_cat, rank_user_num = rank_user_store.lookup(user_ids_arr)
    rank_ctx_cat, rank_ctx_num = encode_rank_context_features(
        tab=tab_arr,
        date=date_arr,
        hourmin=hourmin_arr,
        time_ms=time_ms_arr,
        bucket_sizes=rank_bucket_sizes,
    )
    rank_hist_ids, rank_hist_author_ids, rank_hist_tag_ids, rank_hist_mask = _encode_rank_history_features(
        raw_histories=click_histories,
        item_store=rank_item_store,
        bucket_sizes=rank_bucket_sizes,
        max_history_len=max_history_len,
    )

    recall_model.eval()
    rank_model.eval()
    with torch.no_grad():
        user_vecs = recall_model.encode_user(
            user_cat=torch.from_numpy(recall_user_cat).to(device),
            user_num=torch.from_numpy(recall_user_num).to(device),
            ctx_cat=torch.from_numpy(recall_ctx_cat).to(device),
            hist_ids=torch.from_numpy(hist_ids).to(device),
            hist_mask=torch.from_numpy(hist_mask).to(device),
        )
        recall_preds = batched_topk_inner_product_search(
            query_vecs=user_vecs,
            item_vecs=candidate_item_vecs,
            item_ids=candidate_item_ids,
            topk=route_source_topn,
            query_batch_size=query_batch_size,
            show_progress=show_progress,
            progress_desc="Eval Recall Search",
        )

        content_pred_lists: List[List[int]] = [[] for _ in active_samples]
        if (
            content_route_mode == "zero_shot"
            and content_assets is not None
            and content_budget > 0
            and len(content_assets.item_ids) > 0
        ):
            content_user_vecs, content_valid = build_content_user_vectors(
                samples=active_samples,
                content_assets=content_assets,
                history_len=content_history_len,
                strong_weight=content_strong_weight,
                weak_weight=content_weak_weight,
                decay_half_life_hours=content_decay_half_life_hours,
                device=device,
            )
            valid_idx = np.flatnonzero(content_valid)
            if len(valid_idx) > 0:
                content_preds = batched_topk_inner_product_search(
                    query_vecs=content_user_vecs[torch.from_numpy(valid_idx).to(device=device, dtype=torch.long)],
                    item_vecs=content_assets.item_vecs,
                    item_ids=content_assets.item_ids,
                    topk=route_source_topn,
                    query_batch_size=query_batch_size,
                    show_progress=show_progress,
                    progress_desc="Eval Content Search",
                )
                for pos, sample_idx in enumerate(valid_idx.tolist()):
                    content_pred_lists[sample_idx] = content_preds[pos].tolist()
        elif (
            content_route_mode == "learned_twotower"
            and learned_content_assets is not None
            and content_budget > 0
        ):
            content_pred_lists = _build_learned_content_candidate_lists(
                eval_samples=active_samples,
                content_model=learned_content_assets["model"],
                content_cfg=learned_content_assets["config"],
                content_store=learned_content_assets["content_store"],
                category_store=learned_content_assets["category_store"],
                device=device,
                candidate_topn=route_source_topn,
                item_batch_size=learned_content_assets["item_batch_size"],
                query_batch_size=query_batch_size,
                show_progress=show_progress,
            )

        hstu_pred_lists, hstu_query_count = build_hstu_candidate_lists(
            samples=active_samples,
            assets=hstu_assets,
            topn=route_source_topn,
            device=device,
            query_batch_size=query_batch_size,
            show_progress=show_progress,
            progress_desc="Eval HSTU Search",
        )

        iterator = tqdm(
            enumerate(active_samples),
            total=len(active_samples),
            desc="Eval Rerank",
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for sample_idx, sample in iterator:
            gt = sample.positives
            recall_pred = recall_preds[sample_idx].tolist()
            main_route_candidates = recall_pred[:route_source_topn]
            content_route_candidates = content_pred_lists[sample_idx][:route_source_topn]
            hstu_route_candidates = hstu_pred_lists[sample_idx][:route_source_topn]
            rerank_candidates = np.asarray(
                _build_fused_candidates(
                    fusion_method=fusion_method,
                    main_candidates=main_route_candidates,
                    content_candidates=content_route_candidates,
                    hstu_candidates=hstu_route_candidates,
                    main_budget=main_budget,
                    content_budget=content_budget,
                    hstu_budget=hstu_budget,
                    rrf_k=rrf_k,
                ),
                dtype=np.int64,
            )
            main_candidate_total += float(len(main_route_candidates))
            content_candidate_total += float(len(content_route_candidates))
            hstu_candidate_total += float(len(hstu_route_candidates))
            recall_candidate_total += float(len(rerank_candidates))

            for k in topk:
                route_main_totals[f"route_main_hr@{k}"] += hit_rate_at_k(main_route_candidates, gt, k)
                route_main_totals[f"route_main_ndcg@{k}"] += ndcg_at_k(main_route_candidates, gt_items=gt, k=k)
                route_content_totals[f"route_content_hr@{k}"] += hit_rate_at_k(content_route_candidates, gt, k)
                route_content_totals[f"route_content_ndcg@{k}"] += ndcg_at_k(content_route_candidates, gt_items=gt, k=k)
                route_hstu_totals[f"route_hstu_hr@{k}"] += hit_rate_at_k(hstu_route_candidates, gt, k)
                route_hstu_totals[f"route_hstu_ndcg@{k}"] += ndcg_at_k(hstu_route_candidates, gt_items=gt, k=k)
                totals[f"recall_hr@{k}"] += hit_rate_at_k(rerank_candidates.tolist(), gt, k)
                totals[f"recall_ndcg@{k}"] += ndcg_at_k(rerank_candidates.tolist(), gt_items=gt, k=k)

            # Re-rank recall candidates using shared-bottom scores.
            item_cat, item_num, valid = rank_item_store.lookup(rerank_candidates.astype(np.int64))
            if not np.any(valid):
                skipped += 1
                continue
            item_cat = item_cat[valid]
            item_num = item_num[valid]
            rerank_item_ids = rerank_candidates[valid]

            rep_n = len(rerank_item_ids)
            user_cat_rep = np.repeat(rank_user_cat[sample_idx : sample_idx + 1], repeats=rep_n, axis=0)
            user_num_rep = np.repeat(rank_user_num[sample_idx : sample_idx + 1], repeats=rep_n, axis=0)
            ctx_cat_rep = np.repeat(rank_ctx_cat[sample_idx : sample_idx + 1], repeats=rep_n, axis=0)
            ctx_num_rep = np.repeat(rank_ctx_num[sample_idx : sample_idx + 1], repeats=rep_n, axis=0)
            hist_ids_rep = np.repeat(rank_hist_ids[sample_idx : sample_idx + 1], repeats=rep_n, axis=0)
            hist_author_rep = np.repeat(rank_hist_author_ids[sample_idx : sample_idx + 1], repeats=rep_n, axis=0)
            hist_tag_rep = np.repeat(rank_hist_tag_ids[sample_idx : sample_idx + 1], repeats=rep_n, axis=0)
            hist_mask_rep = np.repeat(rank_hist_mask[sample_idx : sample_idx + 1], repeats=rep_n, axis=0)

            click_logit = rank_model(
                user_cat=torch.from_numpy(user_cat_rep).to(device),
                user_num=torch.from_numpy(user_num_rep).to(device),
                ctx_cat=torch.from_numpy(ctx_cat_rep).to(device),
                ctx_num=torch.from_numpy(ctx_num_rep).to(device),
                hist_ids=torch.from_numpy(hist_ids_rep).to(device),
                hist_author_ids=torch.from_numpy(hist_author_rep).to(device),
                hist_tag_ids=torch.from_numpy(hist_tag_rep).to(device),
                hist_mask=torch.from_numpy(hist_mask_rep).to(device),
                item_cat=torch.from_numpy(item_cat).to(device),
                item_num=torch.from_numpy(item_num).to(device),
            )
            click_prob = torch.sigmoid(click_logit).cpu().numpy()
            final_score = score_click_weight * click_prob
            order = np.argsort(-final_score)
            reranked = rerank_item_ids[order].tolist()

            for k in topk:
                totals[f"e2e_hr@{k}"] += hit_rate_at_k(reranked, gt_items=gt, k=k)
                totals[f"e2e_ndcg@{k}"] += ndcg_at_k(reranked, gt_items=gt, k=k)

    used = len(active_samples)
    unique_users = len({sample.user_id for sample in active_samples})
    out = {k: (v / used if used > 0 else 0.0) for k, v in totals.items()}
    out["num_users"] = float(used)
    out["num_eval_points"] = float(used)
    out["num_unique_users"] = float(unique_users)
    out["num_skipped"] = float(skipped)
    out["num_candidates"] = float(len(candidate_item_ids))
    out["avg_recall_candidates"] = float(recall_candidate_total / used if used > 0 else 0.0)
    out["avg_main_candidates"] = float(main_candidate_total / used if used > 0 else 0.0)
    out["avg_content_candidates"] = float(content_candidate_total / used if used > 0 else 0.0)
    out["avg_hstu_candidates"] = float(hstu_candidate_total / used if used > 0 else 0.0)
    out["num_hstu_queries"] = float(hstu_query_count)
    out["fusion_total_budget"] = float(main_budget + content_budget + hstu_budget)
    out.update({k: (v / used if used > 0 else 0.0) for k, v in route_main_totals.items()})
    out.update({k: (v / used if used > 0 else 0.0) for k, v in route_content_totals.items()})
    out.update({k: (v / used if used > 0 else 0.0) for k, v in route_hstu_totals.items()})
    return out


@torch.no_grad()
def evaluate_ranking_auc(
    rank_model: DINDCNRanker,
    rank_loader: DataLoader,
    device: torch.device,
    show_progress: bool,
) -> Dict[str, float]:
    rank_model.eval()
    click_labels = []
    click_scores = []
    user_ids = []

    iterator = tqdm(
        rank_loader,
        total=len(rank_loader),
        desc="Eval Rank AUC",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in iterator:
        batch = _move_batch_to_device(batch, device=device)
        valid_mask = batch["valid_item_mask"] > 0.5
        if valid_mask.sum().item() == 0:
            continue
        idx = valid_mask.nonzero(as_tuple=True)[0]

        click_logit = rank_model(
            user_cat=batch["user_cat"][idx],
            user_num=batch["user_num"][idx],
            ctx_cat=batch["ctx_cat"][idx],
            ctx_num=batch["ctx_num"][idx],
            hist_ids=batch["hist_ids"][idx],
            hist_author_ids=batch["hist_author_ids"][idx],
            hist_tag_ids=batch["hist_tag_ids"][idx],
            hist_mask=batch["hist_mask"][idx],
            item_cat=batch["item_cat"][idx],
            item_num=batch["item_num"][idx],
        )
        click_prob = torch.sigmoid(click_logit).detach().cpu().numpy()

        click_scores.append(click_prob)
        click_labels.append(batch["label_click"][idx].detach().cpu().numpy())
        user_ids.append(batch["user_ids"][idx].detach().cpu().numpy())

    if not click_scores:
        return {"auc_click": 0.0, "gauc_click": 0.0, "logloss_click": 0.0, "num_rows": 0.0}

    y_click = np.concatenate(click_labels, axis=0)
    y_click_score = np.concatenate(click_scores, axis=0)
    user_ids_arr = np.concatenate(user_ids, axis=0)
    return {
        "auc_click": _safe_auc(y_click, y_click_score),
        "gauc_click": _safe_gauc(user_ids_arr, y_click, y_click_score),
        "logloss_click": _safe_logloss(y_click, y_click_score),
        "num_rows": float(len(y_click)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[Eval] warning: CUDA requested but unavailable, falling back to CPU.")

    topk = parse_topk(args.topk)
    if not topk:
        raise ValueError("topk is empty")

    processed_dir = args.processed_dir
    test_path = processed_dir / "interactions.test.csv"
    train_path = processed_dir / "interactions.train.csv"
    user_path = processed_dir / "user_features.selected.csv"
    item_path = processed_dir / "item_features.selected.csv"

    print("[Eval] loading checkpoints ...")
    recall_model, recall_buckets = _load_recall_model(args.recall_ckpt, device=device)
    rank_model, rank_buckets = _load_rank_model(args.rank_ckpt, device=device)
    if recall_buckets != rank_buckets:
        print("[Eval] warning: recall/rank bucket sizes differ, using route-specific encoders.")

    print("[Eval] loading train/test interactions ...")
    train_df = load_interactions(
        csv_path=train_path,
        max_rows=None,
        sample_frac=1.0,
        positive_only_mode=None,
        seed=args.seed,
    )
    test_df = load_interactions(
        csv_path=test_path,
        max_rows=args.test_max_rows,
        sample_frac=1.0,
        positive_only_mode=None,
        seed=args.seed,
    )
    print(f"[Eval] test rows={len(test_df)}")

    candidate_ids = _build_candidate_ids(
        candidate_item_ids_path=args.candidate_item_ids,
        fallback_item_path=item_path,
    )
    test_ids = np.unique(test_df["video_id"].to_numpy(dtype=np.int64))
    union_ids = np.unique(np.concatenate([candidate_ids, test_ids]))

    print("[Eval] building feature stores ...")
    recall_user_store = UserFeatureStore.from_csv(user_path, bucket_sizes=recall_buckets)
    recall_item_store = ItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=recall_buckets,
        candidate_video_ids=union_ids,
    )
    rank_user_store = RankUserFeatureStore.from_csv(user_path, bucket_sizes=rank_buckets)
    rank_item_store = RankItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=rank_buckets,
        candidate_video_ids=union_ids,
    )
    print("[Eval] encoding recall candidates ...")
    candidate_ids, candidate_vecs = _encode_candidate_item_vectors(
        model=recall_model,
        item_store=recall_item_store,
        candidate_ids=candidate_ids,
        device=device,
        item_batch_size=args.item_batch_size,
        show_progress=not args.disable_progress,
    )
    content_assets = None
    content_route_mode = "none"
    learned_content_assets = None
    if args.content_recall_topn > 0:
        if args.content_item_emb is None or args.content_video_id_to_index is None:
            raise ValueError(
                "content recall is enabled but --content-item-emb/--content-video-id-to-index were not provided."
            )
        if args.content_ckpt is not None:
            content_model, content_cfg, _ = _load_content_model(args.content_ckpt, device=device)
            content_store = load_content_embedding_store(
                embedding_path=args.content_item_emb,
                video_id_to_index_path=args.content_video_id_to_index,
                candidate_item_ids=union_ids,
            )
            category_store = load_category_feature_store(
                category_asset_path=args.content_category_csv,
                raw_category_csv=args.raw_category_csv,
                candidate_item_ids=content_store.item_ids,
            )
            learned_content_assets = {
                "model": content_model,
                "config": content_cfg,
                "content_store": content_store,
                "category_store": category_store,
                "item_batch_size": args.item_batch_size,
            }
            content_route_mode = "learned_twotower"
            print(f"[Eval] learned content candidate items={len(content_store.item_ids)}")
        else:
            content_assets = load_content_recall_assets(
                embedding_path=args.content_item_emb,
                video_id_to_index_path=args.content_video_id_to_index,
                candidate_item_ids=candidate_ids,
                device=device,
            )
            content_route_mode = "zero_shot"
            print(f"[Eval] zero-shot content candidate items={len(content_assets.item_ids)}")
    hstu_assets = None
    if args.hstu_recall_topn > 0:
        if args.hstu_ckpt is None or args.hstu_data_dir is None:
            raise ValueError("HSTU recall is enabled but --hstu-ckpt/--hstu-data-dir were not provided.")
        hstu_assets = load_hstu_route_assets(
            ckpt_path=args.hstu_ckpt,
            data_dir=args.hstu_data_dir,
            candidate_video_ids=candidate_ids,
            split_name="test",
            device=device,
        )
        print(
            f"[Eval] hstu candidate items={len(hstu_assets.candidate_video_ids)} "
            f"eval rows={len(hstu_assets.eval_lookup)}"
        )

    eval_users = build_pointwise_eval_samples(
        interactions=test_df,
        positive_label_mode=args.positive_label_mode,
        max_samples=args.max_eval_users,
    )
    print(f"[Eval] eval points={len(eval_users)}, candidate items={len(candidate_ids)}")

    stage_metrics = evaluate_recall_and_rerank(
        recall_model=recall_model,
        rank_model=rank_model,
        eval_samples=eval_users,
        recall_user_store=recall_user_store,
        recall_item_store=recall_item_store,
        rank_user_store=rank_user_store,
        rank_item_store=rank_item_store,
        candidate_item_ids=candidate_ids,
        candidate_item_vecs=candidate_vecs,
        recall_bucket_sizes=recall_buckets,
        rank_bucket_sizes=rank_buckets,
        topk=topk,
        recall_topn=args.recall_topn,
        max_history_len=args.max_history_len,
        score_click_weight=args.score_click_weight,
        score_long_weight=args.score_long_weight,
        score_like_weight=args.score_like_weight,
        device=device,
        query_batch_size=args.query_batch_size,
        skip_no_candidate_positive=args.skip_no_candidate_positive,
        show_progress=not args.disable_progress,
        content_assets=content_assets,
        content_route_mode=content_route_mode,
        learned_content_assets=learned_content_assets,
        content_recall_topn=args.content_recall_topn,
        content_history_len=args.content_history_len,
        content_strong_weight=args.content_strong_weight,
        content_weak_weight=args.content_weak_weight,
        content_decay_half_life_hours=args.content_decay_half_life_hours,
        hstu_assets=hstu_assets,
        hstu_recall_topn=args.hstu_recall_topn,
        fusion_method=args.fusion_method,
        route_source_topn=args.route_source_topn,
        rrf_k=args.rrf_k,
    )

    # AUC on observed exposures in test split.
    _, _, keep = rank_item_store.lookup(test_df["video_id"].to_numpy(dtype=np.int64))
    test_df_auc = test_df[keep].reset_index(drop=True)
    rank_ds = RankTrainDataset(test_df_auc)
    collator = RankBatchCollator(
        user_store=rank_user_store,
        item_store=rank_item_store,
        bucket_sizes=rank_buckets,
        max_history_len=args.max_history_len,
        device=torch.device("cpu"),
    )
    rank_loader_kwargs = {
        "batch_size": args.rank_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        rank_loader_kwargs["persistent_workers"] = True
        rank_loader_kwargs["prefetch_factor"] = 4
    rank_loader = DataLoader(
        rank_ds,
        **rank_loader_kwargs,
    )
    auc_metrics = evaluate_ranking_auc(
        rank_model=rank_model,
        rank_loader=rank_loader,
        device=device,
        show_progress=not args.disable_progress,
    )

    result = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "recall_rerank_metrics": stage_metrics,
        "ranking_auc_metrics": auc_metrics,
        "test_rows": int(len(test_df)),
        "test_rows_used_for_auc": int(len(test_df_auc)),
    }
    save_json(args.output_json, result)
    print(f"[Eval] metrics saved to {args.output_json}")
    print(result)


if __name__ == "__main__":
    main()
