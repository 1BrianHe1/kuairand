#!/usr/bin/env python3
"""One-click runner for KuaiRand baseline pipeline.

Stages:
1) (optional) rebuild processed data
2) train recall two-tower
3) train shared-bottom ranker
4) evaluate end-to-end metrics
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
from pathlib import Path
from typing import List

from data_utils import load_json


def _run(cmd: List[str]) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[run_all] $ {pretty}", flush=True)
    subprocess.run(cmd, check=True)


def _append_optional_arg(cmd: List[str], flag: str, value: object) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _default_run_name() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="One-click run for KuaiRand baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/KuaiRand-Pure/data"),
        help="Raw KuaiRand data directory (used if --rebuild-data).",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="pure",
        choices=["auto", "1k", "pure"],
        help="Raw dataset variant passed to dataset.py when --rebuild-data is enabled.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=base_dir / "processed_pure",
        help="Processed data directory.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=base_dir / "checkpoints" / "runs",
        help="Root directory for run artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=_default_run_name(),
        help="Run folder name under runs-root.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild processed data before training.",
    )
    parser.add_argument("--max-history-len", type=int, default=500)
    parser.add_argument("--content-history-len", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # recall
    parser.add_argument("--recall-epochs", type=int, default=2)
    parser.add_argument("--recall-batch-size", type=int, default=512)
    parser.add_argument("--recall-train-max-rows", type=int, default=None)
    parser.add_argument("--recall-valid-max-rows", type=int, default=None)
    parser.add_argument(
        "--recall-label-mode",
        type=str,
        default="click_or_long",
        choices=["click", "long_view", "click_or_long", "signal_positive"],
    )
    parser.add_argument("--eval-max-users", type=int, default=None)
    parser.add_argument("--recall-topn", type=int, default=300)
    parser.add_argument("--content-recall-topn", type=int, default=0)
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
        default=None,
    )
    parser.add_argument(
        "--raw-category-csv",
        type=Path,
        default=None,
    )
    parser.add_argument("--content-strong-weight", type=float, default=1.0)
    parser.add_argument("--content-weak-weight", type=float, default=0.5)
    parser.add_argument("--content-decay-half-life-hours", type=float, default=48.0)
    parser.add_argument("--hstu-recall-topn", type=int, default=0)
    parser.add_argument("--hstu-ckpt", type=Path, default=None)
    parser.add_argument("--hstu-data-dir", type=Path, default=None)
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="weighted_rrf",
        choices=["weighted_rrf"],
    )
    parser.add_argument("--route-source-topn", type=int, default=200)
    parser.add_argument("--rrf-k", type=float, default=60.0)

    # rank
    parser.add_argument("--rank-epochs", type=int, default=2)
    parser.add_argument("--rank-batch-size", type=int, default=512)
    parser.add_argument("--rank-train-max-rows", type=int, default=None)
    parser.add_argument("--rank-valid-max-rows", type=int, default=None)

    # evaluation
    parser.add_argument("--test-max-rows", type=int, default=None)
    parser.add_argument("--topk", type=str, default="20,50")
    parser.add_argument("--score-click-weight", type=float, default=0.5)
    parser.add_argument("--score-long-weight", type=float, default=0.5)
    parser.add_argument("--score-like-weight", type=float, default=0.0)
    parser.add_argument(
        "--positive-label-mode",
        type=str,
        default="click_or_long",
        choices=["click", "long_view", "click_or_long", "signal_positive"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use small-row/1-epoch config for quick sanity run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    if args.smoke:
        args.recall_epochs = 1
        args.rank_epochs = 1
        args.recall_train_max_rows = 30_000 if args.recall_train_max_rows is None else min(args.recall_train_max_rows, 30_000)
        args.recall_valid_max_rows = 20_000 if args.recall_valid_max_rows is None else min(args.recall_valid_max_rows, 20_000)
        args.rank_train_max_rows = 50_000 if args.rank_train_max_rows is None else min(args.rank_train_max_rows, 50_000)
        args.rank_valid_max_rows = 20_000 if args.rank_valid_max_rows is None else min(args.rank_valid_max_rows, 20_000)
        args.test_max_rows = 20_000 if args.test_max_rows is None else min(args.test_max_rows, 20_000)
        args.eval_max_users = 50 if args.eval_max_users is None else min(args.eval_max_users, 50)

    run_dir = args.runs_root / args.run_name
    recall_dir = run_dir / "recall"
    rank_dir = run_dir / "rank"
    eval_dir = run_dir / "eval"
    eval_json = eval_dir / "test_metrics.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    python_exec = "python"

    if args.rebuild_data:
        _run(
            [
                python_exec,
                str(base_dir / "dataset.py"),
                "--data-dir",
                str(args.data_dir),
                "--dataset-version",
                str(args.dataset_version),
                "--output-dir",
                str(args.processed_dir),
                "--max-history-len",
                str(args.max_history_len),
                "--content-history-len",
                str(args.content_history_len),
            ]
        )

    recall_cmd = [
        python_exec,
        str(base_dir / "train_recall_twotower.py"),
        "--processed-dir",
        str(args.processed_dir),
        "--output-dir",
        str(recall_dir),
        "--epochs",
        str(args.recall_epochs),
        "--batch-size",
        str(args.recall_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--label-mode",
        str(args.recall_label_mode),
        "--max-history-len",
        str(args.max_history_len),
        "--topk",
        str(args.topk),
        "--recall-topn",
        str(args.recall_topn),
        "--content-recall-topn",
        str(args.content_recall_topn),
        "--content-history-len",
        str(args.content_history_len),
        "--content-strong-weight",
        str(args.content_strong_weight),
        "--content-weak-weight",
        str(args.content_weak_weight),
        "--content-decay-half-life-hours",
        str(args.content_decay_half_life_hours),
        "--hstu-recall-topn",
        str(args.hstu_recall_topn),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
    ]
    _append_optional_arg(recall_cmd, "--content-item-emb", args.content_item_emb)
    _append_optional_arg(recall_cmd, "--content-video-id-to-index", args.content_video_id_to_index)
    _append_optional_arg(recall_cmd, "--hstu-ckpt", args.hstu_ckpt)
    _append_optional_arg(recall_cmd, "--hstu-data-dir", args.hstu_data_dir)
    _append_optional_arg(recall_cmd, "--train-max-rows", args.recall_train_max_rows)
    _append_optional_arg(recall_cmd, "--valid-max-rows", args.recall_valid_max_rows)
    _append_optional_arg(recall_cmd, "--max-eval-users", args.eval_max_users)
    _run(recall_cmd)

    rank_cmd = [
        python_exec,
        str(base_dir / "train_rank_shared_bottom.py"),
        "--processed-dir",
        str(args.processed_dir),
        "--output-dir",
        str(rank_dir),
        "--epochs",
        str(args.rank_epochs),
        "--batch-size",
        str(args.rank_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-history-len",
        str(args.max_history_len),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
    ]
    _append_optional_arg(rank_cmd, "--train-max-rows", args.rank_train_max_rows)
    _append_optional_arg(rank_cmd, "--valid-max-rows", args.rank_valid_max_rows)
    _run(rank_cmd)

    eval_cmd = [
        python_exec,
        str(base_dir / "evaluate_pipeline.py"),
        "--processed-dir",
        str(args.processed_dir),
        "--recall-ckpt",
        str(recall_dir / "recall_model.pt"),
        "--rank-ckpt",
        str(rank_dir / "rank_model.pt"),
        "--candidate-item-ids",
        str(recall_dir / "candidate_item_ids.npy"),
        "--output-json",
        str(eval_json),
        "--topk",
        str(args.topk),
        "--num-workers",
        str(args.num_workers),
        "--recall-topn",
        str(args.recall_topn),
        "--content-recall-topn",
        str(args.content_recall_topn),
        "--content-history-len",
        str(args.content_history_len),
        "--content-strong-weight",
        str(args.content_strong_weight),
        "--content-weak-weight",
        str(args.content_weak_weight),
        "--content-decay-half-life-hours",
        str(args.content_decay_half_life_hours),
        "--hstu-recall-topn",
        str(args.hstu_recall_topn),
        "--fusion-method",
        str(args.fusion_method),
        "--route-source-topn",
        str(args.route_source_topn),
        "--rrf-k",
        str(args.rrf_k),
        "--max-history-len",
        str(args.max_history_len),
        "--score-click-weight",
        str(args.score_click_weight),
        "--score-long-weight",
        str(args.score_long_weight),
        "--score-like-weight",
        str(args.score_like_weight),
        "--positive-label-mode",
        str(args.positive_label_mode),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
    ]
    _append_optional_arg(eval_cmd, "--content-item-emb", args.content_item_emb)
    _append_optional_arg(eval_cmd, "--content-video-id-to-index", args.content_video_id_to_index)
    _append_optional_arg(eval_cmd, "--content-ckpt", args.content_ckpt)
    _append_optional_arg(eval_cmd, "--content-category-csv", args.content_category_csv)
    _append_optional_arg(eval_cmd, "--raw-category-csv", args.raw_category_csv)
    _append_optional_arg(eval_cmd, "--hstu-ckpt", args.hstu_ckpt)
    _append_optional_arg(eval_cmd, "--hstu-data-dir", args.hstu_data_dir)
    _append_optional_arg(eval_cmd, "--test-max-rows", args.test_max_rows)
    _append_optional_arg(eval_cmd, "--max-eval-users", args.eval_max_users)
    _run(eval_cmd)

    metrics = load_json(eval_json)
    recall_metrics = metrics.get("recall_rerank_metrics", {})
    rank_auc = metrics.get("ranking_auc_metrics", {})
    print("\n[run_all] finished", flush=True)
    print(f"[run_all] run_dir: {run_dir}", flush=True)
    print(f"[run_all] eval_json: {eval_json}", flush=True)
    print(
        "[run_all] key metrics: "
        f"recall_hr@50={recall_metrics.get('recall_hr@50', 0.0):.6f}, "
        f"e2e_ndcg@50={recall_metrics.get('e2e_ndcg@50', 0.0):.6f}, "
        f"auc_click={rank_auc.get('auc_click', 0.0):.6f}, "
        f"auc_long_view={rank_auc.get('auc_long_view', 0.0):.6f}, "
        f"auc_like={rank_auc.get('auc_like', 0.0):.6f}"
    , flush=True)


if __name__ == "__main__":
    main()
