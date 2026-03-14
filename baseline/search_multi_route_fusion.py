#!/usr/bin/env python3
"""Search fixed-budget multi-route fusion settings for KuaiRand recall."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from compare_recall_hit_overlap import (
    _build_candidate_ids,
    _build_content_candidate_lists,
    _build_main_candidate_lists,
    _load_content_model,
    _load_main_model,
)
from data_utils import (
    DEFAULT_BUCKET_SIZES,
    ItemFeatureStore,
    UserFeatureStore,
    build_pointwise_eval_samples,
    hit_rate_at_k,
    load_interactions,
    ndcg_at_k,
    parse_topk,
    save_json,
    set_seed,
)
from hstu_route_utils import build_hstu_candidate_lists, load_hstu_route_assets
from train_recall_content_twotower import load_category_feature_store, load_content_embedding_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search fixed-budget multi-route fusion.")
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
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/recall_route_overlap/fusion_search.json"),
    )
    parser.add_argument("--topk", type=str, default="50,100,200")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="valid")
    parser.add_argument("--valid-max-rows", type=int, default=None)
    parser.add_argument("--test-max-rows", type=int, default=None)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--route-topn", type=int, default=200)
    parser.add_argument("--budget-total", type=int, default=200)
    parser.add_argument("--budget-step", type=int, default=20)
    parser.add_argument("--min-main-budget", type=int, default=20)
    parser.add_argument("--min-content-budget", type=int, default=20)
    parser.add_argument("--min-hstu-budget", type=int, default=20)
    parser.add_argument("--methods", type=str, default="weighted_rrf")
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--item-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--max-history-len", type=int, default=500)
    parser.add_argument(
        "--positive-label-mode",
        choices=["click", "long_view", "click_or_long", "signal_positive"],
        default="click_or_long",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def _iter_budget_configs(
    total: int,
    step: int,
    min_main: int,
    min_content: int,
    min_hstu: int,
) -> Iterable[Tuple[int, int, int]]:
    seen = set()
    for main_budget in range(min_main, total + 1, step):
        for content_budget in range(min_content, total + 1, step):
            hstu_budget = total - main_budget - content_budget
            if hstu_budget < min_hstu:
                continue
            if hstu_budget % step != 0:
                continue
            key = (main_budget, content_budget, hstu_budget)
            if key in seen:
                continue
            seen.add(key)
            yield key


def _fuse_weighted_rrf(
    route_lists: Dict[str, Sequence[int]],
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


def _evaluate_fused_lists(
    pred_lists: Sequence[Sequence[int]],
    samples,
    topk: List[int],
) -> Dict[str, float]:
    totals = {f"hr@{k}": 0.0 for k in topk}
    totals.update({f"ndcg@{k}": 0.0 for k in topk})
    candidate_total = 0.0
    used = len(samples)
    for pred, sample in zip(pred_lists, samples):
        gt = sample.positives
        candidate_total += float(len(pred))
        for k in topk:
            totals[f"hr@{k}"] += hit_rate_at_k(pred, gt_items=gt, k=k)
            totals[f"ndcg@{k}"] += ndcg_at_k(pred, gt_items=gt, k=k)
    if used == 0:
        out = {key: 0.0 for key in totals}
        out["avg_candidates"] = 0.0
        return out
    out = {key: value / used for key, value in totals.items()}
    out["avg_candidates"] = candidate_total / used
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    topk = parse_topk(args.topk)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if not topk:
        raise ValueError("topk is empty")
    if not methods:
        raise ValueError("methods is empty")
    if any(method != "weighted_rrf" for method in methods):
        raise ValueError("Only weighted_rrf is supported.")

    processed_dir = args.processed_dir
    eval_path = processed_dir / f"interactions.{args.split}.csv"
    user_path = processed_dir / "user_features.selected.csv"
    item_path = processed_dir / "item_features.selected.csv"

    max_rows = args.valid_max_rows if args.split == "valid" else args.test_max_rows
    eval_df = load_interactions(eval_path, max_rows=max_rows, seed=args.seed)
    eval_samples = build_pointwise_eval_samples(
        interactions=eval_df,
        positive_label_mode=args.positive_label_mode,
        max_samples=args.max_eval_users,
    )

    main_model, bucket_sizes = _load_main_model(args.main_ckpt, device=device)
    candidate_ids = _build_candidate_ids(args.main_candidate_item_ids, item_path)
    test_ids = (
        np.unique(eval_df["video_id"].to_numpy(dtype=np.int64))
        if len(eval_df) > 0
        else np.zeros((0,), dtype=np.int64)
    )
    union_ids = np.unique(np.concatenate([candidate_ids, test_ids])) if len(test_ids) > 0 else candidate_ids

    user_store = UserFeatureStore.from_csv(user_path, bucket_sizes=bucket_sizes)
    item_store = ItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=bucket_sizes,
        candidate_video_ids=union_ids,
    )

    content_model, content_cfg, _ = _load_content_model(args.content_ckpt, device=device)
    content_store = load_content_embedding_store(
        embedding_path=args.content_item_emb,
        video_id_to_index_path=args.content_video_id_to_index,
        candidate_item_ids=None,
    )
    category_store = load_category_feature_store(
        category_asset_path=args.content_category_csv,
        raw_category_csv=args.raw_category_csv,
        candidate_item_ids=content_store.item_ids,
    )

    hstu_assets = load_hstu_route_assets(
        ckpt_path=args.hstu_ckpt,
        data_dir=args.hstu_data_dir,
        candidate_video_ids=union_ids,
        split_name=args.split,
        device=device,
    )

    main_pred_lists = _build_main_candidate_lists(
        samples=eval_samples,
        model=main_model,
        user_store=user_store,
        item_store=item_store,
        bucket_sizes=bucket_sizes,
        device=device,
        candidate_topn=args.route_topn,
        max_history_len=args.max_history_len,
        item_batch_size=args.item_batch_size,
        query_batch_size=args.query_batch_size,
        show_progress=not args.disable_progress,
    )
    content_pred_lists = _build_content_candidate_lists(
        samples=eval_samples,
        model=content_model,
        ckpt_cfg=content_cfg,
        content_store=content_store,
        category_store=category_store,
        device=device,
        candidate_topn=args.route_topn,
        item_batch_size=args.item_batch_size,
        query_batch_size=args.query_batch_size,
        show_progress=not args.disable_progress,
    )
    hstu_pred_lists, num_hstu_queries = build_hstu_candidate_lists(
        samples=eval_samples,
        assets=hstu_assets,
        topn=args.route_topn,
        device=device,
        query_batch_size=args.query_batch_size,
        show_progress=not args.disable_progress,
        progress_desc="HSTU Recall Search",
    )

    results = []
    for main_budget, content_budget, hstu_budget in _iter_budget_configs(
        total=args.budget_total,
        step=args.budget_step,
        min_main=args.min_main_budget,
        min_content=args.min_content_budget,
        min_hstu=args.min_hstu_budget,
    ):
        route_weights = {
            "main": float(main_budget) / float(args.budget_total),
            "content": float(content_budget) / float(args.budget_total),
            "hstu": float(hstu_budget) / float(args.budget_total),
        }
        trimmed_main = [pred[:main_budget] for pred in main_pred_lists]
        trimmed_content = [pred[:content_budget] for pred in content_pred_lists]
        trimmed_hstu = [pred[:hstu_budget] for pred in hstu_pred_lists]

        for method in methods:
            fused_pred_lists: List[List[int]] = []
            for idx in range(len(eval_samples)):
                fused = _fuse_weighted_rrf(
                    {
                        "hstu": trimmed_hstu[idx],
                        "main": trimmed_main[idx],
                        "content": trimmed_content[idx],
                    },
                    route_weights=route_weights,
                    total_budget=args.budget_total,
                    rrf_k=args.rrf_k,
                )
                fused_pred_lists.append(fused)

            metrics = _evaluate_fused_lists(
                pred_lists=fused_pred_lists,
                samples=eval_samples,
                topk=topk,
            )
            results.append(
                {
                    "method": method,
                    "main_budget": int(main_budget),
                    "content_budget": int(content_budget),
                    "hstu_budget": int(hstu_budget),
                    "route_weights": route_weights,
                    "metrics": metrics,
                }
            )

    results_sorted = sorted(
        results,
        key=lambda row: (
            -row["metrics"].get(f"hr@{topk[-1]}", 0.0),
            -row["metrics"].get(f"ndcg@{topk[-1]}", 0.0),
            -row["metrics"].get(f"hr@{topk[0]}", 0.0),
        ),
    )

    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "split": args.split,
        "num_split_rows": int(len(eval_df)),
        "num_eval_points": int(len(eval_samples)),
        "num_hstu_queries": int(num_hstu_queries),
        "best_by_hr_at_max_k": results_sorted[0] if results_sorted else None,
        "top_results": results_sorted[:10],
        "all_results": results_sorted,
    }
    save_json(args.output_json, summary)
    print(f"[FusionSearch] result saved to {args.output_json}")
    if results_sorted:
        print(results_sorted[0])


if __name__ == "__main__":
    main()
