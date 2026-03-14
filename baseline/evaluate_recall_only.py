#!/usr/bin/env python3
"""Recall-only evaluation for KuaiRand routes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_utils import (
    DEFAULT_BUCKET_SIZES,
    ItemFeatureStore,
    UserFeatureStore,
    build_pointwise_eval_samples,
    load_content_recall_assets,
    load_interactions,
    parse_topk,
    read_ids_from_npy,
    save_json,
    set_seed,
)
from hstu_route_utils import load_hstu_route_assets
from models import TwoTowerRecallModel
from train_recall_twotower import evaluate_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recall-only evaluation for KuaiRand routes.")
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
        "--candidate-item-ids",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/recall_pure/candidate_item_ids.npy"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/eval_recall_only_pure/recall_metrics.json"),
    )
    parser.add_argument("--topk", type=str, default="50,100,200")
    parser.add_argument("--valid-max-rows", type=int, default=None)
    parser.add_argument("--test-max-rows", type=int, default=None)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--recall-topn", type=int, default=200)
    parser.add_argument("--content-recall-topn", type=int, default=0)
    parser.add_argument("--content-item-emb", type=Path, default=None)
    parser.add_argument("--content-video-id-to-index", type=Path, default=None)
    parser.add_argument("--content-history-len", type=int, default=5)
    parser.add_argument("--content-strong-weight", type=float, default=1.0)
    parser.add_argument("--content-weak-weight", type=float, default=0.5)
    parser.add_argument("--content-decay-half-life-hours", type=float, default=48.0)
    parser.add_argument("--hstu-recall-topn", type=int, default=0)
    parser.add_argument("--hstu-ckpt", type=Path, default=None)
    parser.add_argument("--hstu-data-dir", type=Path, default=None)
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
    parser.add_argument("--skip-no-candidate-positive", action="store_true")
    return parser.parse_args()


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


def _build_candidate_ids(candidate_item_ids_path: Path, fallback_item_path: Path) -> np.ndarray:
    if candidate_item_ids_path.exists():
        candidate_ids = read_ids_from_npy(candidate_item_ids_path)
    else:
        item_ids = pd.read_csv(fallback_item_path, usecols=["video_id"])["video_id"].to_numpy(dtype=np.int64)
        candidate_ids = np.unique(item_ids)
    return np.unique(candidate_ids.astype(np.int64))


def _evaluate_split(
    split_name: str,
    interactions: pd.DataFrame,
    model: TwoTowerRecallModel,
    user_store: UserFeatureStore,
    item_store: ItemFeatureStore,
    bucket_sizes: dict,
    args: argparse.Namespace,
    device: torch.device,
    candidate_ids: np.ndarray,
    content_assets,
):
    eval_samples = build_pointwise_eval_samples(
        interactions=interactions,
        positive_label_mode=args.positive_label_mode,
        max_samples=args.max_eval_users,
    )
    hstu_assets = None
    if args.hstu_recall_topn > 0:
        if args.hstu_ckpt is None or args.hstu_data_dir is None:
            raise ValueError("HSTU recall is enabled but --hstu-ckpt/--hstu-data-dir were not provided.")
        hstu_assets = load_hstu_route_assets(
            ckpt_path=args.hstu_ckpt,
            data_dir=args.hstu_data_dir,
            candidate_video_ids=candidate_ids,
            split_name=split_name,
            device=device,
        )
    metrics = evaluate_recall(
        model=model,
        user_store=user_store,
        item_store=item_store,
        eval_samples=eval_samples,
        topk=parse_topk(args.topk),
        bucket_sizes=bucket_sizes,
        max_history_len=args.max_history_len,
        device=device,
        recall_topn=args.recall_topn,
        item_batch_size=args.item_batch_size,
        query_batch_size=args.query_batch_size,
        skip_no_candidate_positive=args.skip_no_candidate_positive,
        show_progress=not args.disable_progress,
        content_assets=content_assets,
        content_recall_topn=args.content_recall_topn,
        content_history_len=args.content_history_len,
        content_strong_weight=args.content_strong_weight,
        content_weak_weight=args.content_weak_weight,
        content_decay_half_life_hours=args.content_decay_half_life_hours,
        hstu_assets=hstu_assets,
        hstu_recall_topn=args.hstu_recall_topn,
    )
    return {
        "num_rows": int(len(interactions)),
        "num_eval_points": int(len(eval_samples)),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    processed_dir = args.processed_dir
    valid_path = processed_dir / "interactions.valid.csv"
    test_path = processed_dir / "interactions.test.csv"
    user_path = processed_dir / "user_features.selected.csv"
    item_path = processed_dir / "item_features.selected.csv"

    recall_model, bucket_sizes = _load_recall_model(args.recall_ckpt, device=device)
    valid_df = load_interactions(valid_path, max_rows=args.valid_max_rows, seed=args.seed)
    test_df = load_interactions(test_path, max_rows=args.test_max_rows, seed=args.seed)

    candidate_ids = _build_candidate_ids(args.candidate_item_ids, item_path)
    valid_ids = np.unique(valid_df["video_id"].to_numpy(dtype=np.int64)) if len(valid_df) > 0 else np.zeros((0,), dtype=np.int64)
    test_ids = np.unique(test_df["video_id"].to_numpy(dtype=np.int64)) if len(test_df) > 0 else np.zeros((0,), dtype=np.int64)
    union_ids = np.unique(np.concatenate([candidate_ids, valid_ids, test_ids]))

    user_store = UserFeatureStore.from_csv(user_path, bucket_sizes=bucket_sizes)
    item_store = ItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=bucket_sizes,
        candidate_video_ids=union_ids,
    )

    content_assets = None
    if args.content_recall_topn > 0:
        if args.content_item_emb is None or args.content_video_id_to_index is None:
            raise ValueError(
                "content recall is enabled but --content-item-emb/--content-video-id-to-index were not provided."
            )
        content_assets = load_content_recall_assets(
            embedding_path=args.content_item_emb,
            video_id_to_index_path=args.content_video_id_to_index,
            candidate_item_ids=candidate_ids,
            device=device,
        )

    result = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "valid": _evaluate_split(
            split_name="valid",
            interactions=valid_df,
            model=recall_model,
            user_store=user_store,
            item_store=item_store,
            bucket_sizes=bucket_sizes,
            args=args,
            device=device,
            candidate_ids=candidate_ids,
            content_assets=content_assets,
        ),
        "test": _evaluate_split(
            split_name="test",
            interactions=test_df,
            model=recall_model,
            user_store=user_store,
            item_store=item_store,
            bucket_sizes=bucket_sizes,
            args=args,
            device=device,
            candidate_ids=candidate_ids,
            content_assets=content_assets,
        ),
    }
    save_json(args.output_json, result)
    print(f"[RecallOnly] metrics saved to {args.output_json}")
    print(result)


if __name__ == "__main__":
    main()
