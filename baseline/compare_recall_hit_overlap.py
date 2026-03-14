#!/usr/bin/env python3
"""Compare recall routes by unique hit overlap and merge gain on KuaiRand."""

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
    load_interactions,
    parse_topk,
    read_ids_from_npy,
    save_json,
    set_seed,
)
from hstu_route_utils import build_hstu_candidate_lists, load_hstu_route_assets
from models import ContentTwoTowerRecallModel, TwoTowerRecallModel
from train_recall_content_twotower import (
    _build_candidate_item_bank as _build_content_candidate_item_bank,
    _build_eval_query_features,
    load_category_feature_store,
    load_content_embedding_store,
)
from train_recall_twotower import _build_candidate_item_bank as _build_main_candidate_item_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare main/content/HSTU recall routes by hit overlap and merge gain."
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
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/recall_route_overlap/test_overlap.json"),
    )
    parser.add_argument("--topk", type=str, default="50,100,200")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-progress", action="store_true")
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
    if len(samples) == 0:
        return []
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
    item_batch_size: int,
    query_batch_size: int,
    show_progress: bool,
) -> List[List[int]]:
    if len(samples) == 0:
        return []
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
        top_items = batched_topk_inner_product_search(
            query_vecs=user_vecs,
            item_vecs=cand_item_vecs,
            item_ids=cand_item_ids,
            topk=candidate_topn,
            query_batch_size=query_batch_size,
            show_progress=show_progress,
            progress_desc="Content Recall Search",
        )
    return [row.tolist() for row in top_items]


def _route_hit_flags(pred_lists: List[List[int]], samples, k: int) -> np.ndarray:
    flags = np.zeros((len(samples),), dtype=bool)
    for idx, (pred, sample) in enumerate(zip(pred_lists, samples)):
        top_items = pred[:k]
        flags[idx] = any(int(item_id) in sample.positives for item_id in top_items)
    return flags


def _count_rate(mask: np.ndarray) -> dict:
    count = int(mask.sum())
    total = int(len(mask))
    return {
        "count": count,
        "rate": float(count / total) if total > 0 else 0.0,
    }


def _compute_overlap_and_gain(
    main_hits: np.ndarray,
    content_hits: np.ndarray,
    hstu_hits: np.ndarray,
) -> dict:
    main_only = main_hits & ~content_hits & ~hstu_hits
    content_only = ~main_hits & content_hits & ~hstu_hits
    hstu_only = ~main_hits & ~content_hits & hstu_hits
    main_content_only = main_hits & content_hits & ~hstu_hits
    main_hstu_only = main_hits & ~content_hits & hstu_hits
    content_hstu_only = ~main_hits & content_hits & hstu_hits
    all_three = main_hits & content_hits & hstu_hits

    union_main_content = main_hits | content_hits
    union_main_hstu = main_hits | hstu_hits
    union_content_hstu = content_hits | hstu_hits
    union_all = main_hits | content_hits | hstu_hits

    main_hr = float(main_hits.mean()) if len(main_hits) > 0 else 0.0
    content_hr = float(content_hits.mean()) if len(content_hits) > 0 else 0.0
    hstu_hr = float(hstu_hits.mean()) if len(hstu_hits) > 0 else 0.0

    def _merge_entry(mask: np.ndarray, members: List[float], baseline: float | None = None) -> dict:
        union_hr = float(mask.mean()) if len(mask) > 0 else 0.0
        entry = {
            "union_hr": union_hr,
            "gain_vs_best_single": float(union_hr - max(members)) if members else 0.0,
        }
        if baseline is not None:
            entry["gain_vs_baseline"] = float(union_hr - baseline)
        return entry

    return {
        "single_route_hr": {
            "main": main_hr,
            "content": content_hr,
            "hstu": hstu_hr,
        },
        "hit_overlap": {
            "main_only": _count_rate(main_only),
            "content_only": _count_rate(content_only),
            "hstu_only": _count_rate(hstu_only),
            "main_content_only": _count_rate(main_content_only),
            "main_hstu_only": _count_rate(main_hstu_only),
            "content_hstu_only": _count_rate(content_hstu_only),
            "all_three": _count_rate(all_three),
            "main_content_overlap": _count_rate(main_hits & content_hits),
            "main_hstu_overlap": _count_rate(main_hits & hstu_hits),
            "content_hstu_overlap": _count_rate(content_hits & hstu_hits),
            "any_hit": _count_rate(union_all),
        },
        "merge_gain": {
            "main_plus_content": _merge_entry(
                union_main_content,
                [main_hr, content_hr],
                baseline=main_hr,
            ),
            "main_plus_hstu": _merge_entry(
                union_main_hstu,
                [main_hr, hstu_hr],
                baseline=main_hr,
            ),
            "content_plus_hstu": _merge_entry(
                union_content_hstu,
                [content_hr, hstu_hr],
            ),
            "all_three": _merge_entry(
                union_all,
                [main_hr, content_hr, hstu_hr],
                baseline=main_hr,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    topk = parse_topk(args.topk)
    if not topk:
        raise ValueError("topk is empty")

    processed_dir = args.processed_dir
    test_path = processed_dir / "interactions.test.csv"
    user_path = processed_dir / "user_features.selected.csv"
    item_path = processed_dir / "item_features.selected.csv"

    test_df = load_interactions(test_path, max_rows=args.test_max_rows, seed=args.seed)
    eval_samples = build_pointwise_eval_samples(
        interactions=test_df,
        positive_label_mode=args.positive_label_mode,
        max_samples=args.max_eval_users,
    )

    main_model, bucket_sizes = _load_main_model(args.main_ckpt, device=device)
    candidate_ids = _build_candidate_ids(args.main_candidate_item_ids, item_path)
    test_ids = np.unique(test_df["video_id"].to_numpy(dtype=np.int64)) if len(test_df) > 0 else np.zeros((0,), dtype=np.int64)
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
        split_name="test",
        device=device,
    )

    main_pred_lists = _build_main_candidate_lists(
        samples=eval_samples,
        model=main_model,
        user_store=user_store,
        item_store=item_store,
        bucket_sizes=bucket_sizes,
        device=device,
        candidate_topn=args.main_topn,
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
        candidate_topn=args.content_topn,
        item_batch_size=args.item_batch_size,
        query_batch_size=args.query_batch_size,
        show_progress=not args.disable_progress,
    )
    hstu_pred_lists, num_hstu_queries = build_hstu_candidate_lists(
        samples=eval_samples,
        assets=hstu_assets,
        topn=args.hstu_topn,
        device=device,
        query_batch_size=args.query_batch_size,
        show_progress=not args.disable_progress,
        progress_desc="HSTU Recall Search",
    )

    result = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "test_rows": int(len(test_df)),
        "num_eval_points": int(len(eval_samples)),
        "num_hstu_queries": int(num_hstu_queries),
        "per_k": {},
    }
    for k in topk:
        main_hits = _route_hit_flags(main_pred_lists, eval_samples, k=k)
        content_hits = _route_hit_flags(content_pred_lists, eval_samples, k=k)
        hstu_hits = _route_hit_flags(hstu_pred_lists, eval_samples, k=k)
        result["per_k"][str(k)] = _compute_overlap_and_gain(main_hits, content_hits, hstu_hits)

    save_json(args.output_json, result)
    print(f"[RecallOverlap] result saved to {args.output_json}")
    print(result)


if __name__ == "__main__":
    main()
