#!/usr/bin/env python3
"""Recall-only evaluation for main/content/HSTU routes on KuaiRand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from data_utils import (
    DEFAULT_BUCKET_SIZES,
    ItemFeatureStore,
    UserFeatureStore,
    batched_topk_inner_product_search,
    build_pointwise_eval_samples,
    encode_context_features,
    encode_history_batch,
    hit_rate_at_k,
    load_interactions,
    ndcg_at_k,
    parse_history_field,
    parse_topk,
    read_ids_from_npy,
    save_json,
    set_seed,
)
from hstu_route_utils import build_hstu_candidate_lists, load_hstu_route_assets
from models import ContentTwoTowerRecallModel, TwoTowerRecallModel
from train_recall_content_twotower import (
    CategoryFeatureStore,
    _build_candidate_item_bank as _build_content_candidate_item_bank,
    _build_eval_query_features,
    compute_content_immunity_metrics,
    load_category_feature_store,
    load_content_embedding_store,
)
from train_recall_twotower import _build_candidate_item_bank as _build_main_candidate_item_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recall-only evaluation for main/content/HSTU routes."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure"),
    )
    parser.add_argument(
        "--main-ckpt",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/recall_pure/recall_model.pt"),
    )
    parser.add_argument(
        "--main-candidate-item-ids",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/recall_pure/candidate_item_ids.npy"),
    )
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
        "--hstu-ckpt",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/hstu_recall_pure/hstu_recall_model.pt"),
    )
    parser.add_argument(
        "--hstu-data-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/hstu_interleaved_firsttoken_len100"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/eval_recall_three_routes_pure/recall_metrics.json"),
    )
    parser.add_argument("--topk", type=str, default="50,100,200")
    parser.add_argument("--valid-max-rows", type=int, default=None)
    parser.add_argument("--test-max-rows", type=int, default=None)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--main-topn", type=int, default=200)
    parser.add_argument("--content-topn", type=int, default=200)
    parser.add_argument("--hstu-topn", type=int, default=200)
    parser.add_argument("--item-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--max-history-len", type=int, default=500)
    parser.add_argument(
        "--positive-label-mode",
        choices=["click", "long_view", "click_or_long", "signal_positive"],
        default="click_or_long",
    )
    parser.add_argument(
        "--fusion-method",
        choices=["weighted_rrf"],
        default="weighted_rrf",
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument(
        "--content-negative-filter-mode",
        choices=["off", "soft", "hard", "hybrid"],
        default="off",
    )
    parser.add_argument("--content-negative-filter-pool-topn", type=int, default=400)
    parser.add_argument("--content-negative-filter-history-len", type=int, default=5)
    parser.add_argument("--content-negative-filter-half-life-hours", type=float, default=48.0)
    parser.add_argument("--content-negative-filter-soft-beta", type=float, default=0.2)
    parser.add_argument("--content-negative-filter-hard-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--skip-no-candidate-positive", action="store_true")
    return parser.parse_args()


def _load_main_model(ckpt_path: Path, device: torch.device) -> tuple[TwoTowerRecallModel, dict]:
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


def _load_content_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[ContentTwoTowerRecallModel, dict]:
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
    return model, cfg


def _build_candidate_ids(candidate_item_ids_path: Path, fallback_item_path: Path) -> np.ndarray:
    if candidate_item_ids_path.exists():
        candidate_ids = read_ids_from_npy(candidate_item_ids_path)
    else:
        item_ids = pd.read_csv(fallback_item_path, usecols=["video_id"])["video_id"].to_numpy(dtype=np.int64)
        candidate_ids = np.unique(item_ids)
    return np.unique(candidate_ids.astype(np.int64))


def _build_main_candidate_lists(
    samples,
    model: TwoTowerRecallModel,
    user_store: UserFeatureStore,
    item_store: ItemFeatureStore,
    bucket_sizes: dict,
    device: torch.device,
    candidate_topn: int,
    max_history_len: int,
    item_batch_size: int,
    query_batch_size: int,
    show_progress: bool,
) -> List[List[int]]:
    if len(samples) == 0 or candidate_topn <= 0:
        return [[] for _ in samples]
    cand_item_ids, cand_item_vecs = _build_main_candidate_item_bank(
        model=model,
        item_store=item_store,
        device=device,
        batch_size=item_batch_size,
        show_progress=show_progress,
    )
    user_ids = np.asarray([sample.user_id for sample in samples], dtype=np.int64)
    tab = np.asarray([sample.tab for sample in samples], dtype=np.int64)
    hour = np.asarray([sample.hour_bucket for sample in samples], dtype=np.int64)
    date = np.asarray([sample.date_bucket for sample in samples], dtype=np.int64)
    histories = [sample.history for sample in samples]
    user_cat, user_num = user_store.lookup(user_ids)
    ctx_cat = encode_context_features(tab, hour, date, bucket_sizes)
    hist_ids, hist_mask = encode_history_batch(
        histories,
        max_history_len=max_history_len,
        item_bucket_size=bucket_sizes["video_id"],
    )
    with torch.no_grad():
        user_vecs = model.encode_user(
            user_cat=torch.from_numpy(user_cat).to(device),
            user_num=torch.from_numpy(user_num).to(device),
            ctx_cat=torch.from_numpy(ctx_cat).to(device),
            hist_ids=torch.from_numpy(hist_ids).to(device),
            hist_mask=torch.from_numpy(hist_mask).to(device),
        )
        top_items = batched_topk_inner_product_search(
            query_vecs=user_vecs,
            item_vecs=cand_item_vecs,
            item_ids=cand_item_ids,
            topk=candidate_topn,
            query_batch_size=query_batch_size,
            show_progress=show_progress,
            progress_desc="Main Recall Search",
        )
    return [row.tolist() for row in top_items]


def _build_content_candidate_lists(
    samples,
    model: ContentTwoTowerRecallModel,
    ckpt_cfg: dict,
    content_store,
    category_store,
    device: torch.device,
    candidate_topn: int,
    filter_mode: str,
    filter_pool_topn: int,
    filter_history_len: int,
    filter_half_life_hours: float,
    filter_soft_beta: float,
    filter_hard_threshold: float,
    item_batch_size: int,
    query_batch_size: int,
    show_progress: bool,
) -> List[List[int]]:
    if len(samples) == 0 or candidate_topn <= 0:
        return [[] for _ in samples]
    cand_item_ids, cand_item_vecs = _build_content_candidate_item_bank(
        model=model,
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
        samples=samples,
        content_store=content_store,
        category_store=category_store,
        short_history_len=int(ckpt_cfg.get("short_history_len", 15)),
        long_term_history_len=int(ckpt_cfg.get("long_term_history_len", 100)),
        negative_history_len=int(ckpt_cfg.get("negative_history_len", 10)),
        short_signal_weight_map={
            1: float(ckpt_cfg.get("click_weight", 0.5)),
            2: float(ckpt_cfg.get("long_view_weight", 1.0)),
            3: float(ckpt_cfg.get("like_weight", 1.25)),
            4: float(ckpt_cfg.get("social_weight", 1.5)),
        },
        long_signal_weight_map={
            2: float(ckpt_cfg.get("long_view_weight", 1.0)),
            3: float(ckpt_cfg.get("like_weight", 1.25)),
            4: float(ckpt_cfg.get("social_weight", 1.5)),
        },
        negative_signal_weight_map={
            1: float(ckpt_cfg.get("negative_relative_weight", 0.35)),
            2: float(ckpt_cfg.get("negative_hate_weight", 1.0)),
        },
        negative_semantic_similarity_threshold=float(
            ckpt_cfg.get("negative_semantic_sim_threshold", 0.1)
        ),
        short_half_life_ms=max(float(ckpt_cfg.get("short_history_half_life_hours", 48.0)), 0.0)
        * 3600.0
        * 1000.0,
        negative_half_life_ms=max(float(ckpt_cfg.get("negative_half_life_hours", 72.0)), 0.0)
        * 3600.0
        * 1000.0,
    )
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
        retrieval_topn = int(candidate_topn)
        if filter_mode != "off":
            retrieval_topn = max(int(candidate_topn), int(filter_pool_topn))
        top_items = batched_topk_inner_product_search(
            query_vecs=user_vecs,
            item_vecs=cand_item_vecs,
            item_ids=cand_item_ids,
            topk=retrieval_topn,
            query_batch_size=query_batch_size,
            show_progress=show_progress,
            progress_desc="Content Recall Search",
        )
    if filter_mode == "off":
        return [row[:candidate_topn].tolist() for row in top_items]
    return _apply_content_negative_filter(
        samples=samples,
        ranked_item_ids=top_items,
        user_vecs=user_vecs,
        candidate_item_ids=cand_item_ids,
        candidate_item_vecs=cand_item_vecs,
        final_topn=candidate_topn,
        mode=filter_mode,
        history_len=filter_history_len,
        half_life_hours=filter_half_life_hours,
        soft_beta=filter_soft_beta,
        hard_threshold=filter_hard_threshold,
    )


def _extract_recent_hate_events(sample, max_history_len: int) -> tuple[np.ndarray, np.ndarray]:
    if max_history_len <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    video_ids = parse_history_field(sample.negative_history_video_ids, max_history_len=0)
    signal_types = parse_history_field(sample.negative_history_signal_types, max_history_len=0)
    event_times = parse_history_field(sample.negative_history_time_ms, max_history_len=0)
    take = min(len(video_ids), len(signal_types), len(event_times))
    if take <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    kept_ids: List[int] = []
    kept_times: List[int] = []
    for idx in range(take):
        if int(signal_types[idx]) != 2:
            continue
        kept_ids.append(int(video_ids[idx]))
        kept_times.append(int(event_times[idx]))
        if len(kept_ids) >= max_history_len:
            break
    if not kept_ids:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(kept_ids, dtype=np.int64), np.asarray(kept_times, dtype=np.int64)


def _lookup_sorted_indices(sorted_ids: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(sorted_ids) == 0 or len(query_ids) == 0:
        return np.zeros((len(query_ids),), dtype=np.int64), np.zeros((len(query_ids),), dtype=bool)
    idx = np.searchsorted(sorted_ids, query_ids)
    idx_clip = np.clip(idx, 0, len(sorted_ids) - 1)
    valid = (idx < len(sorted_ids)) & (sorted_ids[idx_clip] == query_ids)
    out = np.zeros((len(query_ids),), dtype=np.int64)
    if np.any(valid):
        out[valid] = idx[valid]
    return out, valid


def _apply_content_negative_filter(
    *,
    samples,
    ranked_item_ids: np.ndarray,
    user_vecs: torch.Tensor,
    candidate_item_ids: np.ndarray,
    candidate_item_vecs: torch.Tensor,
    final_topn: int,
    mode: str,
    history_len: int,
    half_life_hours: float,
    soft_beta: float,
    hard_threshold: float,
) -> List[List[int]]:
    half_life_ms = max(float(half_life_hours), 0.0) * 3600.0 * 1000.0
    out: List[List[int]] = []
    with torch.no_grad():
        for sample_idx, sample in enumerate(samples):
            row_ids = np.asarray(ranked_item_ids[sample_idx].tolist(), dtype=np.int64)
            if len(row_ids) == 0:
                out.append([])
                continue
            neg_ids, neg_times = _extract_recent_hate_events(sample, max_history_len=history_len)
            if len(neg_ids) == 0:
                out.append(row_ids[:final_topn].tolist())
                continue

            row_item_idx, row_valid = _lookup_sorted_indices(candidate_item_ids, row_ids)
            if not np.all(row_valid):
                row_ids = row_ids[row_valid]
                row_item_idx = row_item_idx[row_valid]
            if len(row_ids) == 0:
                out.append([])
                continue

            neg_item_idx, neg_valid = _lookup_sorted_indices(candidate_item_ids, neg_ids)
            if not np.any(neg_valid):
                out.append(row_ids[:final_topn].tolist())
                continue
            neg_item_idx = neg_item_idx[neg_valid]
            neg_times = neg_times[neg_valid]

            candidate_vec = candidate_item_vecs[row_item_idx]
            base_scores = torch.matmul(candidate_vec, user_vecs[sample_idx])
            neg_vec = candidate_item_vecs[neg_item_idx]
            neg_sim = torch.matmul(candidate_vec, neg_vec.T)

            if half_life_ms > 0.0:
                age_ms = np.maximum(0.0, float(sample.time_ms) - neg_times.astype(np.float32))
                decay = np.power(0.5, age_ms / half_life_ms).astype(np.float32)
            else:
                decay = np.ones((len(neg_times),), dtype=np.float32)
            decay_t = torch.from_numpy(decay).to(neg_sim.device)
            neg_penalty = torch.max(neg_sim * decay_t.unsqueeze(0), dim=1).values

            if mode == "soft":
                rerank_scores = base_scores - float(soft_beta) * neg_penalty
                order = torch.argsort(rerank_scores, descending=True)
                out.append(row_ids[order.cpu().numpy()[:final_topn]].tolist())
                continue

            keep_mask = neg_penalty <= float(hard_threshold)
            kept_ids = row_ids[keep_mask.cpu().numpy()]
            kept_scores = base_scores[keep_mask]
            kept_penalty = neg_penalty[keep_mask]

            if mode == "hard":
                if len(kept_ids) == 0:
                    out.append([])
                    continue
                order = torch.argsort(kept_scores, descending=True)
                out.append(kept_ids[order.cpu().numpy()[:final_topn]].tolist())
                continue

            if len(kept_ids) == 0:
                out.append([])
                continue
            rerank_scores = kept_scores - float(soft_beta) * kept_penalty
            order = torch.argsort(rerank_scores, descending=True)
            out.append(kept_ids[order.cpu().numpy()[:final_topn]].tolist())
    return out


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


def _empty_metrics(topk: List[int], num_candidates: int = 0, skipped: int = 0) -> dict:
    out = {f"hr@{k}": 0.0 for k in topk}
    out.update({f"ndcg@{k}": 0.0 for k in topk})
    out.update({f"route_main_hr@{k}": 0.0 for k in topk})
    out.update({f"route_main_ndcg@{k}": 0.0 for k in topk})
    out.update({f"route_content_hr@{k}": 0.0 for k in topk})
    out.update({f"route_content_ndcg@{k}": 0.0 for k in topk})
    out.update({f"route_hstu_hr@{k}": 0.0 for k in topk})
    out.update({f"route_hstu_ndcg@{k}": 0.0 for k in topk})
    out["num_users"] = 0.0
    out["num_eval_points"] = 0.0
    out["num_unique_users"] = 0.0
    out["num_skipped"] = float(skipped)
    out["num_candidates"] = float(num_candidates)
    out["avg_recall_candidates"] = 0.0
    out["avg_main_candidates"] = 0.0
    out["avg_content_candidates"] = 0.0
    out["avg_hstu_candidates"] = 0.0
    out["num_hstu_queries"] = 0.0
    out.update(
        compute_content_immunity_metrics(
            pred_lists=[],
            samples=[],
            category_store=CategoryFeatureStore(
                item_ids=np.zeros((0,), dtype=np.int64),
                category_l1_index=np.zeros((0,), dtype=np.int64),
                category_l2_index=np.zeros((0,), dtype=np.int64),
                num_category_l1=0,
                num_category_l2=0,
            ),
            topk=topk,
            negative_history_len=0,
            prefix="route_content_",
        )
    )
    return out


def _evaluate_split(
    *,
    split_name: str,
    interactions: pd.DataFrame,
    topk: List[int],
    candidate_ids: np.ndarray,
    main_model: TwoTowerRecallModel,
    main_bucket_sizes: dict,
    user_store: UserFeatureStore,
    item_store: ItemFeatureStore,
    content_model: ContentTwoTowerRecallModel | None,
    content_cfg: dict | None,
    content_store,
    category_store,
    hstu_ckpt: Path | None,
    hstu_data_dir: Path | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    eval_samples = build_pointwise_eval_samples(
        interactions=interactions,
        positive_label_mode=args.positive_label_mode,
        max_samples=args.max_eval_users,
        max_negative_history_len=max(
            200,
            int(content_cfg.get("negative_history_len", 10)) if content_cfg is not None else 0,
        ),
    )
    if not eval_samples:
        return {
            "num_rows": int(len(interactions)),
            "num_eval_points": 0,
            "metrics": _empty_metrics(topk, num_candidates=len(candidate_ids)),
        }

    skipped = 0
    cand_set = set(candidate_ids.tolist())
    active_samples = []
    for sample in eval_samples:
        if args.skip_no_candidate_positive and len(sample.positives.intersection(cand_set)) == 0:
            skipped += 1
            continue
        active_samples.append(sample)

    if not active_samples:
        return {
            "num_rows": int(len(interactions)),
            "num_eval_points": int(len(eval_samples)),
            "metrics": _empty_metrics(topk, num_candidates=len(candidate_ids), skipped=skipped),
        }

    main_source_topn = max(max(topk), int(args.main_topn), 1)
    content_source_topn = max(max(topk), int(args.content_topn), 1) if args.content_topn > 0 else 0
    hstu_source_topn = max(max(topk), int(args.hstu_topn), 1) if args.hstu_topn > 0 else 0

    main_pred_lists = _build_main_candidate_lists(
        samples=active_samples,
        model=main_model,
        user_store=user_store,
        item_store=item_store,
        bucket_sizes=main_bucket_sizes,
        device=device,
        candidate_topn=main_source_topn,
        max_history_len=args.max_history_len,
        item_batch_size=args.item_batch_size,
        query_batch_size=args.query_batch_size,
        show_progress=not args.disable_progress,
    )

    content_pred_lists = [[] for _ in active_samples]
    if args.content_topn > 0:
        if content_model is None or content_cfg is None or content_store is None or category_store is None:
            raise ValueError("content route is enabled but content assets are incomplete.")
        content_pred_lists = _build_content_candidate_lists(
            samples=active_samples,
            model=content_model,
            ckpt_cfg=content_cfg,
            content_store=content_store,
            category_store=category_store,
            device=device,
            candidate_topn=content_source_topn,
            filter_mode=args.content_negative_filter_mode,
            filter_pool_topn=args.content_negative_filter_pool_topn,
            filter_history_len=args.content_negative_filter_history_len,
            filter_half_life_hours=args.content_negative_filter_half_life_hours,
            filter_soft_beta=args.content_negative_filter_soft_beta,
            filter_hard_threshold=args.content_negative_filter_hard_threshold,
            item_batch_size=args.item_batch_size,
            query_batch_size=args.query_batch_size,
            show_progress=not args.disable_progress,
        )

    hstu_pred_lists = [[] for _ in active_samples]
    num_hstu_queries = 0
    if args.hstu_topn > 0:
        if hstu_ckpt is None or hstu_data_dir is None:
            raise ValueError("HSTU route is enabled but --hstu-ckpt/--hstu-data-dir were not provided.")
        hstu_assets = load_hstu_route_assets(
            ckpt_path=hstu_ckpt,
            data_dir=hstu_data_dir,
            candidate_video_ids=candidate_ids,
            split_name=split_name,
            device=device,
        )
        hstu_pred_lists, num_hstu_queries = build_hstu_candidate_lists(
            samples=active_samples,
            assets=hstu_assets,
            topn=hstu_source_topn,
            device=device,
            query_batch_size=args.query_batch_size,
            show_progress=not args.disable_progress,
            progress_desc=f"HSTU Recall Search ({split_name})",
        )

    totals = _empty_metrics(topk, num_candidates=len(candidate_ids), skipped=skipped)
    main_candidate_total = 0.0
    content_candidate_total = 0.0
    hstu_candidate_total = 0.0
    recall_candidate_total = 0.0

    for idx, sample in enumerate(active_samples):
        gt = sample.positives
        main_items = main_pred_lists[idx][: args.main_topn]
        content_items = content_pred_lists[idx][: args.content_topn]
        hstu_items = hstu_pred_lists[idx][: args.hstu_topn]
        fused_items = _build_fused_candidates(
            fusion_method=args.fusion_method,
            main_candidates=main_items,
            content_candidates=content_items,
            hstu_candidates=hstu_items,
            main_budget=args.main_topn,
            content_budget=args.content_topn,
            hstu_budget=args.hstu_topn,
            rrf_k=args.rrf_k,
        )

        main_candidate_total += float(len(main_items))
        content_candidate_total += float(len(content_items))
        hstu_candidate_total += float(len(hstu_items))
        recall_candidate_total += float(len(fused_items))
        for k in topk:
            totals[f"route_main_hr@{k}"] += hit_rate_at_k(main_items, gt_items=gt, k=k)
            totals[f"route_main_ndcg@{k}"] += ndcg_at_k(main_items, gt_items=gt, k=k)
            totals[f"route_content_hr@{k}"] += hit_rate_at_k(content_items, gt_items=gt, k=k)
            totals[f"route_content_ndcg@{k}"] += ndcg_at_k(content_items, gt_items=gt, k=k)
            totals[f"route_hstu_hr@{k}"] += hit_rate_at_k(hstu_items, gt_items=gt, k=k)
            totals[f"route_hstu_ndcg@{k}"] += ndcg_at_k(hstu_items, gt_items=gt, k=k)
            totals[f"hr@{k}"] += hit_rate_at_k(fused_items, gt_items=gt, k=k)
            totals[f"ndcg@{k}"] += ndcg_at_k(fused_items, gt_items=gt, k=k)

    used = len(active_samples)
    unique_users = len({sample.user_id for sample in active_samples})
    for key, value in list(totals.items()):
        if key.startswith("hr@") or key.startswith("ndcg@") or key.startswith("route_"):
            totals[key] = float(value / used) if used > 0 else 0.0
    totals["num_users"] = float(used)
    totals["num_eval_points"] = float(used)
    totals["num_unique_users"] = float(unique_users)
    totals["num_skipped"] = float(skipped)
    totals["num_candidates"] = float(len(candidate_ids))
    totals["avg_recall_candidates"] = float(recall_candidate_total / used) if used > 0 else 0.0
    totals["avg_main_candidates"] = float(main_candidate_total / used) if used > 0 else 0.0
    totals["avg_content_candidates"] = float(content_candidate_total / used) if used > 0 else 0.0
    totals["avg_hstu_candidates"] = float(hstu_candidate_total / used) if used > 0 else 0.0
    totals["num_hstu_queries"] = float(num_hstu_queries)
    totals["fusion_total_budget"] = float(args.main_topn + args.content_topn + args.hstu_topn)
    if args.content_topn > 0 and category_store is not None and content_cfg is not None:
        totals.update(
            compute_content_immunity_metrics(
                pred_lists=content_pred_lists,
                samples=active_samples,
                category_store=category_store,
                topk=topk,
                negative_history_len=int(content_cfg.get("negative_history_len", 10)),
                prefix="route_content_",
            )
        )

    return {
        "num_rows": int(len(interactions)),
        "num_eval_points": int(len(eval_samples)),
        "metrics": totals,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[Recall3] warning: CUDA requested but unavailable, falling back to CPU.")

    topk = parse_topk(args.topk)
    if not topk:
        raise ValueError("topk is empty")

    processed_dir = args.processed_dir
    valid_path = processed_dir / "interactions.valid.csv"
    test_path = processed_dir / "interactions.test.csv"
    user_path = processed_dir / "user_features.selected.csv"
    item_path = processed_dir / "item_features.selected.csv"

    valid_df = load_interactions(valid_path, max_rows=args.valid_max_rows, seed=args.seed)
    test_df = load_interactions(test_path, max_rows=args.test_max_rows, seed=args.seed)

    main_model, bucket_sizes = _load_main_model(args.main_ckpt, device=device)
    candidate_ids = _build_candidate_ids(args.main_candidate_item_ids, item_path)
    valid_ids = np.unique(valid_df["video_id"].to_numpy(dtype=np.int64)) if len(valid_df) > 0 else np.zeros((0,), dtype=np.int64)
    test_ids = np.unique(test_df["video_id"].to_numpy(dtype=np.int64)) if len(test_df) > 0 else np.zeros((0,), dtype=np.int64)
    union_ids = np.unique(np.concatenate([candidate_ids, valid_ids, test_ids]))

    user_store = UserFeatureStore.from_csv(user_path, bucket_sizes=bucket_sizes)
    item_store = ItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=bucket_sizes,
        candidate_video_ids=union_ids,
    )

    content_model = None
    content_cfg = None
    content_store = None
    category_store = None
    if args.content_topn > 0:
        content_model, content_cfg = _load_content_model(args.content_ckpt, device=device)
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

    result = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "valid": _evaluate_split(
            split_name="valid",
            interactions=valid_df,
            topk=topk,
            candidate_ids=union_ids,
            main_model=main_model,
            main_bucket_sizes=bucket_sizes,
            user_store=user_store,
            item_store=item_store,
            content_model=content_model,
            content_cfg=content_cfg,
            content_store=content_store,
            category_store=category_store,
            hstu_ckpt=args.hstu_ckpt,
            hstu_data_dir=args.hstu_data_dir,
            args=args,
            device=device,
        ),
        "test": _evaluate_split(
            split_name="test",
            interactions=test_df,
            topk=topk,
            candidate_ids=union_ids,
            main_model=main_model,
            main_bucket_sizes=bucket_sizes,
            user_store=user_store,
            item_store=item_store,
            content_model=content_model,
            content_cfg=content_cfg,
            content_store=content_store,
            category_store=category_store,
            hstu_ckpt=args.hstu_ckpt,
            hstu_data_dir=args.hstu_data_dir,
            args=args,
            device=device,
        ),
    }
    save_json(args.output_json, result)
    print(f"[Recall3] metrics saved to {args.output_json}")
    print(result)


if __name__ == "__main__":
    main()
