#!/usr/bin/env python3
"""Train single-task DIN+DCN ranker for click."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_utils import (
    DEFAULT_RANK_BUCKET_SIZES,
    RankBatchCollator,
    RankItemFeatureStore,
    RankTrainDataset,
    RankUserFeatureStore,
    load_interactions,
    save_json,
    set_seed,
)
from models import DINDCNRanker


def parse_args() -> argparse.Namespace:
    default_num_workers = max(1, min(8, (os.cpu_count() or 4) // 2))
    parser = argparse.ArgumentParser(description="Train KuaiRand single-task DIN+DCN ranker baseline.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/rank_pure"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dcn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-history-len", type=int, default=20)
    parser.add_argument("--train-max-rows", type=int, default=None)
    parser.add_argument("--train-sample-frac", type=float, default=1.0)
    parser.add_argument("--valid-max-rows", type=int, default=None)
    parser.add_argument("--item-max-count", type=int, default=0)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training on CUDA.")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation every N epochs; the final epoch is always evaluated.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    p = np.clip(y_score.astype(np.float64), 1e-7, 1.0 - 1e-7)
    y = y_true.astype(np.float64)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _safe_gauc(user_ids: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(user_ids) == 0:
        return 0.0
    order = np.argsort(user_ids, kind="mergesort")
    user_sorted = user_ids[order]
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    total_auc = 0.0
    total_weight = 0.0
    start = 0
    while start < len(user_sorted):
        end = start + 1
        while end < len(user_sorted) and user_sorted[end] == user_sorted[start]:
            end += 1
        labels = y_true_sorted[start:end]
        if len(np.unique(labels)) >= 2:
            weight = float(end - start)
            total_auc += weight * float(roc_auc_score(labels, y_score_sorted[start:end]))
            total_weight += weight
        start = end
    if total_weight <= 0.0:
        return 0.0
    return float(total_auc / total_weight)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out

# 单任务 click BCE
def train_one_epoch(
    model: DINDCNRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    pos_weight: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    show_progress: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_batches = 0

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=f"Rank Train Epoch {epoch}",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in iterator:
        batch = _move_batch_to_device(batch, device=device)
        valid_mask = batch["valid_item_mask"] > 0.5
        if valid_mask.sum().item() < 2:
            continue
        idx = valid_mask.nonzero(as_tuple=True)[0]

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            click_logit = model(
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
            label_click = batch["label_click"][idx]
            click_loss = F.binary_cross_entropy_with_logits(
                click_logit,
                label_click,
                pos_weight=pos_weight,
            )
            loss = click_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = len(idx)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        total_batches += 1
        if show_progress and total_samples > 0:
            iterator.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    if total_samples == 0:
        return {"loss": 0.0, "samples": 0.0, "batches": 0.0}
    return {
        "loss": total_loss / total_samples,
        "samples": float(total_samples),
        "batches": float(total_batches),
    }


@torch.no_grad()
def evaluate_ranker(
    model: DINDCNRanker,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    show_progress: bool,
) -> Dict[str, float]:
    model.eval()
    y_click = []
    y_click_prob = []
    user_ids = []
    total_loss = 0.0
    total_samples = 0

    iterator = tqdm(
        loader,
        total=len(loader),
        desc="Rank Valid",
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

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            click_logit = model(
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
            label_click = batch["label_click"][idx]
            click_loss = F.binary_cross_entropy_with_logits(
                click_logit,
                label_click,
                pos_weight=pos_weight,
            )
            loss = click_loss

        prob_click = torch.sigmoid(click_logit).detach().cpu().numpy()
        y_click.append(label_click.detach().cpu().numpy())
        y_click_prob.append(prob_click)
        user_ids.append(batch["user_ids"][idx].detach().cpu().numpy())

        bs = len(idx)
        total_loss += float(loss.item()) * bs
        total_samples += bs

    if total_samples == 0:
        return {
            "loss": 0.0,
            "auc_click": 0.0,
            "gauc_click": 0.0,
            "logloss_click": 0.0,
            "samples": 0.0,
        }

    y_click_arr = np.concatenate(y_click, axis=0)
    y_click_prob_arr = np.concatenate(y_click_prob, axis=0)
    user_ids_arr = np.concatenate(user_ids, axis=0)

    auc_click = _safe_auc(y_click_arr, y_click_prob_arr)
    gauc_click = _safe_gauc(user_ids_arr, y_click_arr, y_click_prob_arr)
    logloss_click = _safe_logloss(y_click_arr, y_click_prob_arr)
    return {
        "loss": total_loss / total_samples,
        "auc_click": auc_click,
        "gauc_click": gauc_click,
        "logloss_click": logloss_click,
        "samples": float(total_samples),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = bool(args.amp and device.type == "cuda")

    processed_dir = args.processed_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "interactions.train.csv"
    valid_path = processed_dir / "interactions.valid.csv"
    user_path = processed_dir / "user_features.selected.csv"
    item_path = processed_dir / "item_features.selected.csv"

    print("[Rank] loading interactions ...")
    train_df = load_interactions(
        csv_path=train_path,
        max_rows=args.train_max_rows,
        sample_frac=args.train_sample_frac,
        positive_only_mode=None,
        seed=args.seed,
    )
    valid_df = load_interactions(
        csv_path=valid_path,
        max_rows=args.valid_max_rows,
        sample_frac=1.0,
        positive_only_mode=None,
        seed=args.seed,
    )
    if len(train_df) == 0:
        raise RuntimeError("train_df is empty.")
    print(f"[Rank] train rows={len(train_df)}, valid rows={len(valid_df)}")

    bucket_sizes = dict(DEFAULT_RANK_BUCKET_SIZES)
    user_store = RankUserFeatureStore.from_csv(user_path, bucket_sizes=bucket_sizes)

    item_ids = np.unique(
        np.concatenate(
            [
                train_df["video_id"].to_numpy(dtype=np.int64),
                valid_df["video_id"].to_numpy(dtype=np.int64),
            ]
        )
    )
    max_items = args.item_max_count if args.item_max_count > 0 else None
    item_store = RankItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=bucket_sizes,
        candidate_video_ids=item_ids,
        max_items=max_items,
    )
    print(f"[Rank] item rows={item_store.num_items}")

    # Keep rows with item features.
    _, _, keep_train = item_store.lookup(train_df["video_id"].to_numpy(dtype=np.int64))
    train_df = train_df[keep_train].reset_index(drop=True)
    _, _, keep_valid = item_store.lookup(valid_df["video_id"].to_numpy(dtype=np.int64))
    valid_df = valid_df[keep_valid].reset_index(drop=True)
    print(f"[Rank] train rows after item join={len(train_df)}, valid rows after item join={len(valid_df)}")

    train_ds = RankTrainDataset(train_df)
    valid_ds = RankTrainDataset(valid_df)

    collator = RankBatchCollator(
        user_store=user_store,
        item_store=item_store,
        bucket_sizes=bucket_sizes,
        max_history_len=args.max_history_len,
        device=torch.device("cpu"),
    )
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    valid_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 4
        valid_loader_kwargs["persistent_workers"] = True
        valid_loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_ds,
        **train_loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        **valid_loader_kwargs,
    )

    model = DINDCNRanker(
        bucket_sizes=bucket_sizes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dcn_layers=args.dcn_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    click_sum = float(train_df["is_click"].sum())
    neg_sum = float(len(train_df) - click_sum)
    pos_weight_value = float(args.pos_weight) if args.pos_weight is not None else (neg_sum / max(click_sum, 1.0))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    print(f"[Rank] task=click pos_weight={pos_weight_value:.5f}")

    best_score = -1.0
    best_metrics = {}
    records: List[dict] = []
    best_path = output_dir / "rank_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            pos_weight=pos_weight,
            device=device,
            amp_enabled=amp_enabled,
            show_progress=not args.disable_progress,
            epoch=epoch,
        )
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        valid_stats = {}
        train_elapsed = time.time() - t0
        valid_elapsed = 0.0
        if should_eval:
            t1 = time.time()
            valid_stats = evaluate_ranker(
                model=model,
                loader=valid_loader,
                pos_weight=pos_weight,
                device=device,
                amp_enabled=amp_enabled,
                show_progress=not args.disable_progress,
            )
            valid_elapsed = time.time() - t1
        elapsed = train_elapsed + valid_elapsed
        score = valid_stats.get("auc_click", best_score)

        records.append(
            {
                "epoch": epoch,
                "elapsed_sec": elapsed,
                "train_elapsed_sec": train_elapsed,
                "valid_elapsed_sec": valid_elapsed,
                "train": train_stats,
                "valid": valid_stats,
            }
        )
        if should_eval:
            print(
                f"[Rank][Epoch {epoch}] loss={train_stats['loss']:.5f} "
                f"auc_click={valid_stats['auc_click']:.5f} "
                f"gauc_click={valid_stats['gauc_click']:.5f} "
                f"logloss_click={valid_stats['logloss_click']:.5f} "
                f"train={train_elapsed:.1f}s valid={valid_elapsed:.1f}s"
            )
        else:
            print(
                f"[Rank][Epoch {epoch}] loss={train_stats['loss']:.5f} "
                f"train={train_elapsed:.1f}s valid=skipped"
            )

        if should_eval and score > best_score:
            best_score = score
            best_metrics = valid_stats
            ckpt_config = {}
            for key, value in vars(args).items():
                if isinstance(value, Path):
                    ckpt_config[key] = str(value)
                else:
                    ckpt_config[key] = value
            ckpt_config.update(
                {
                    "rank_model_type": "din_dcn_click",
                    "rank_task": "click",
                    "deep_hidden_dims": [256, 128],
                    "pos_weight_value": pos_weight_value,
                }
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "bucket_sizes": bucket_sizes,
                    "config": ckpt_config,
                },
                best_path,
            )

    summary = {
        "rank_task": "click",
        "best_score": best_score,
        "best_valid_metrics": best_metrics,
        "epochs": records,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "num_items": int(item_store.num_items),
        "pos_weight_value": pos_weight_value,
    }
    save_json(output_dir / "rank_train_summary.json", summary)
    print(f"[Rank] best checkpoint: {best_path}")
    print(f"[Rank] summary saved: {output_dir / 'rank_train_summary.json'}")


if __name__ == "__main__":
    main()
