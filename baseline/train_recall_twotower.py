#!/usr/bin/env python3
"""Train two-tower recall model with query-level sampled negatives."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_utils import (
    DEFAULT_BUCKET_SIZES,
    EvalUserSample,
    ItemFeatureStore,
    RecallHardNegativeSampler,
    RecallQueryBatchCollator,
    RecallTrainQueryDataset,
    UserFeatureStore,
    batched_topk_inner_product_search,
    build_content_user_vectors,
    build_pointwise_eval_samples,
    encode_context_features,
    encode_history_batch,
    hit_rate_at_k,
    load_interactions,
    load_content_recall_assets,
    merge_recall_candidate_lists,
    ndcg_at_k,
    parse_topk,
    save_json,
    set_seed,
    write_ids_to_npy,
)
from hstu_route_utils import build_hstu_candidate_lists, load_hstu_route_assets
from models import TwoTowerRecallModel


def parse_args() -> argparse.Namespace:
    default_num_workers = max(1, min(8, (os.cpu_count() or 4) // 2))
    parser = argparse.ArgumentParser(description="Train KuaiRand two-tower recall baseline.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/recall_pure"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--num-explicit-negatives", type=int, default=4)
    parser.add_argument("--num-hard-negatives", type=int, default=8)
    parser.add_argument("--num-random-negatives", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--tower-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-history-len", type=int, default=500)
    parser.add_argument("--max-positive-items-per-query", type=int, default=4)
    parser.add_argument(
        "--label-mode",
        choices=["click", "long_view", "click_or_long", "signal_positive"],
        default="click_or_long",
    )
    parser.add_argument("--train-max-rows", type=int, default=None)
    parser.add_argument("--train-sample-frac", type=float, default=1.0)
    parser.add_argument("--valid-max-rows", type=int, default=None)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument(
        "--recall-topn",
        type=int,
        default=300,
        help="Number of recall candidates kept before computing metrics or passing to rerank.",
    )
    parser.add_argument(
        "--content-recall-topn",
        type=int,
        default=0,
        help="Additional content-recall candidates merged into validation recall.",
    )
    parser.add_argument("--content-item-emb", type=Path, default=None)
    parser.add_argument("--content-video-id-to-index", type=Path, default=None)
    parser.add_argument("--content-history-len", type=int, default=5)
    parser.add_argument("--content-strong-weight", type=float, default=1.0)
    parser.add_argument("--content-weak-weight", type=float, default=0.5)
    parser.add_argument("--content-decay-half-life-hours", type=float, default=48.0)
    parser.add_argument("--hstu-recall-topn", type=int, default=0)
    parser.add_argument("--hstu-ckpt", type=Path, default=None)
    parser.add_argument("--hstu-data-dir", type=Path, default=None)
    parser.add_argument("--eval-item-batch-size", type=int, default=4096)
    parser.add_argument("--eval-query-batch-size", type=int, default=64)
    parser.add_argument("--topk", type=str, default="20,50")
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
    parser.add_argument(
        "--disable-false-negative-mask",
        action="store_true",
        help="Deprecated for query-level sampled-negative training; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--skip-no-candidate-positive",
        action="store_true",
        help="Skip users with no positive item present in evaluation candidate pool.",
    )
    return parser.parse_args()


def _build_candidate_item_bank(
    model: TwoTowerRecallModel,
    item_store: ItemFeatureStore,
    device: torch.device,
    batch_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    item_ids = item_store.item_ids
    cat = item_store.cat_features
    num = item_store.num_features

    model.eval()
    all_vecs: List[torch.Tensor] = []
    with torch.no_grad():
        starts = range(0, len(item_ids), batch_size)
        iterator = tqdm(
            starts,
            total=(len(item_ids) + batch_size - 1) // batch_size if len(item_ids) > 0 else 0,
            desc="Recall Candidate Encode",
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for start in iterator:
            end = min(len(item_ids), start + batch_size)
            item_cat_t = torch.from_numpy(cat[start:end]).to(device)
            item_num_t = torch.from_numpy(num[start:end]).to(device)
            vec = model.encode_item(item_cat=item_cat_t, item_num=item_num_t).detach()
            all_vecs.append(vec)
    item_vecs = (
        torch.cat(all_vecs, dim=0)
        if all_vecs
        else torch.zeros((0, model.item_mlp[-1].out_features), device=device)
    )
    return item_ids, item_vecs


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _multi_positive_sampled_softmax_loss(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, int, float]:
    valid_mask = candidate_mask > 0.5
    positive_mask = positive_mask > 0.5
    active_mask = positive_mask.any(dim=1) & (valid_mask.sum(dim=1) > positive_mask.sum(dim=1))
    if active_mask.sum().item() == 0:
        return logits.new_zeros(()), 0, 0.0

    masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    pos_logits = masked_logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    log_num = torch.logsumexp(pos_logits[active_mask], dim=1)
    log_denom = torch.logsumexp(masked_logits[active_mask], dim=1)
    loss = -(log_num - log_denom).mean()
    avg_candidates = float(valid_mask[active_mask].sum(dim=1).float().mean().item())
    return loss, int(active_mask.sum().item()), avg_candidates


def evaluate_recall(
    model: TwoTowerRecallModel,
    user_store: UserFeatureStore,
    item_store: ItemFeatureStore,
    eval_samples: List[EvalUserSample],
    topk: List[int],
    bucket_sizes: Dict[str, int],
    max_history_len: int,
    device: torch.device,
    recall_topn: int,
    item_batch_size: int,
    query_batch_size: int,
    skip_no_candidate_positive: bool,
    show_progress: bool,
    content_assets=None,
    content_recall_topn: int = 0,
    content_history_len: int = 5,
    content_strong_weight: float = 1.0,
    content_weak_weight: float = 0.5,
    content_decay_half_life_hours: float = 48.0,
    hstu_assets=None,
    hstu_recall_topn: int = 0,
) -> Dict[str, float]:
    if not eval_samples:
        metrics = {f"hr@{k}": 0.0 for k in topk}
        metrics.update({f"ndcg@{k}": 0.0 for k in topk})
        metrics.update({f"route_main_hr@{k}": 0.0 for k in topk})
        metrics.update({f"route_main_ndcg@{k}": 0.0 for k in topk})
        metrics.update({f"route_content_hr@{k}": 0.0 for k in topk})
        metrics.update({f"route_content_ndcg@{k}": 0.0 for k in topk})
        metrics.update({f"route_hstu_hr@{k}": 0.0 for k in topk})
        metrics.update({f"route_hstu_ndcg@{k}": 0.0 for k in topk})
        metrics["num_users"] = 0.0
        metrics["num_eval_points"] = 0.0
        metrics["num_unique_users"] = 0.0
        metrics["num_skipped"] = 0.0
        metrics["avg_recall_candidates"] = 0.0
        metrics["avg_main_candidates"] = 0.0
        metrics["avg_content_candidates"] = 0.0
        metrics["avg_hstu_candidates"] = 0.0
        metrics["num_hstu_queries"] = 0.0
        return metrics

    cand_item_ids, cand_item_vecs = _build_candidate_item_bank(
        model=model,
        item_store=item_store,
        device=device,
        batch_size=item_batch_size,
        show_progress=show_progress,
    )
    cand_item_set = set(cand_item_ids.tolist())

    max_k = max(max(topk), recall_topn)
    totals = {f"hr@{k}": 0.0 for k in topk}
    totals.update({f"ndcg@{k}": 0.0 for k in topk})
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

    model.eval()
    active_samples: List[EvalUserSample] = []
    user_ids = []
    tab = []
    hour = []
    date = []
    histories = []
    for sample in eval_samples:
        gt = sample.positives
        if skip_no_candidate_positive and len(gt.intersection(cand_item_set)) == 0:
            skipped += 1
            continue
        active_samples.append(sample)
        user_ids.append(sample.user_id)
        tab.append(sample.tab)
        hour.append(sample.hour_bucket)
        date.append(sample.date_bucket)
        histories.append(sample.history)

    if not active_samples:
        out = {k: 0.0 for k in totals}
        out["num_users"] = 0.0
        out["num_eval_points"] = 0.0
        out["num_unique_users"] = 0.0
        out["num_skipped"] = float(skipped)
        out["num_candidates"] = float(len(cand_item_ids))
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

    user_cat, user_num = user_store.lookup(np.asarray(user_ids, dtype=np.int64))
    ctx_cat = encode_context_features(
        np.asarray(tab, dtype=np.int64),
        np.asarray(hour, dtype=np.int64),
        np.asarray(date, dtype=np.int64),
        bucket_sizes,
    )
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
            topk=max_k,
            query_batch_size=query_batch_size,
            show_progress=show_progress,
            progress_desc="Recall Search",
        )

    content_pred_lists: List[List[int]] = [[] for _ in active_samples]
    if (
        content_assets is not None
        and content_recall_topn > 0
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
            content_top_items = batched_topk_inner_product_search(
                query_vecs=content_user_vecs[torch.from_numpy(valid_idx).to(device=device, dtype=torch.long)],
                item_vecs=content_assets.item_vecs,
                item_ids=content_assets.item_ids,
                topk=content_recall_topn,
                query_batch_size=query_batch_size,
                show_progress=show_progress,
                progress_desc="Content Search",
            )
            for pos, sample_idx in enumerate(valid_idx.tolist()):
                content_pred_lists[sample_idx] = content_top_items[pos].tolist()

    hstu_pred_lists, hstu_query_count = build_hstu_candidate_lists(
        samples=active_samples,
        assets=hstu_assets,
        topn=hstu_recall_topn,
        device=device,
        query_batch_size=query_batch_size,
        show_progress=show_progress,
        progress_desc="HSTU Search",
    )

    used = len(active_samples)
    unique_users = len({sample.user_id for sample in active_samples})
    for sample_idx, (sample, pred_items) in enumerate(zip(active_samples, top_items.tolist())):
        gt = sample.positives
        main_items = list(pred_items[:recall_topn])
        content_items = content_pred_lists[sample_idx][:content_recall_topn]
        hstu_items = hstu_pred_lists[sample_idx][:hstu_recall_topn]
        pred_items_eval = merge_recall_candidate_lists(main_items, content_items)
        pred_items_eval = merge_recall_candidate_lists(pred_items_eval, hstu_items)
        main_candidate_total += float(len(main_items))
        content_candidate_total += float(len(content_items))
        hstu_candidate_total += float(len(hstu_items))
        recall_candidate_total += float(len(pred_items_eval))
        for k in topk:
            route_main_totals[f"route_main_hr@{k}"] += hit_rate_at_k(main_items, gt_items=gt, k=k)
            route_main_totals[f"route_main_ndcg@{k}"] += ndcg_at_k(main_items, gt_items=gt, k=k)
            route_content_totals[f"route_content_hr@{k}"] += hit_rate_at_k(content_items, gt_items=gt, k=k)
            route_content_totals[f"route_content_ndcg@{k}"] += ndcg_at_k(content_items, gt_items=gt, k=k)
            route_hstu_totals[f"route_hstu_hr@{k}"] += hit_rate_at_k(hstu_items, gt_items=gt, k=k)
            route_hstu_totals[f"route_hstu_ndcg@{k}"] += ndcg_at_k(hstu_items, gt_items=gt, k=k)
            totals[f"hr@{k}"] += hit_rate_at_k(pred_items_eval, gt_items=gt, k=k)
            totals[f"ndcg@{k}"] += ndcg_at_k(pred_items_eval, gt_items=gt, k=k)

    out = {k: (v / used if used > 0 else 0.0) for k, v in totals.items()}
    out.update({k: (v / used if used > 0 else 0.0) for k, v in route_main_totals.items()})
    out.update({k: (v / used if used > 0 else 0.0) for k, v in route_content_totals.items()})
    out.update({k: (v / used if used > 0 else 0.0) for k, v in route_hstu_totals.items()})
    out["num_users"] = float(used)
    out["num_eval_points"] = float(used)
    out["num_unique_users"] = float(unique_users)
    out["num_skipped"] = float(skipped)
    out["num_candidates"] = float(len(cand_item_ids))
    out["avg_recall_candidates"] = float(recall_candidate_total / used if used > 0 else 0.0)
    out["avg_main_candidates"] = float(main_candidate_total / used if used > 0 else 0.0)
    out["avg_content_candidates"] = float(content_candidate_total / used if used > 0 else 0.0)
    out["avg_hstu_candidates"] = float(hstu_candidate_total / used if used > 0 else 0.0)
    out["num_hstu_queries"] = float(hstu_query_count)
    return out


def train_one_epoch(
    model: TwoTowerRecallModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    temperature: float,
    device: torch.device,
    amp_enabled: bool,
    show_progress: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_queries = 0
    total_avg_candidates = 0.0

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=f"Recall Train Epoch {epoch}",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in iterator:
        batch = _move_batch_to_device(batch, device=device)
        candidate_mask = batch["candidate_mask"] > 0.5
        if candidate_mask.sum().item() == 0:
            continue

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            user_vec = model.encode_user(
                user_cat=batch["user_cat"],
                user_num=batch["user_num"],
                ctx_cat=batch["ctx_cat"],
                hist_ids=batch["hist_ids"],
                hist_mask=batch["hist_mask"],
            )
            flat_mask = candidate_mask.view(-1)
            flat_idx = flat_mask.nonzero(as_tuple=True)[0]
            flat_item_cat = batch["candidate_item_cat"].view(-1, batch["candidate_item_cat"].size(-1))[flat_idx]
            flat_item_num = batch["candidate_item_num"].view(-1, batch["candidate_item_num"].size(-1))[flat_idx]
            flat_item_vec = model.encode_item(item_cat=flat_item_cat, item_num=flat_item_num)
            item_dim = flat_item_vec.size(-1)
            item_vec = torch.zeros(
                batch["candidate_item_cat"].size(0),
                batch["candidate_item_cat"].size(1),
                item_dim,
                device=user_vec.device,
                dtype=flat_item_vec.dtype,
            )
            item_vec.view(-1, item_dim)[flat_idx] = flat_item_vec
            logits = (user_vec.unsqueeze(1) * item_vec).sum(dim=-1) / temperature
            loss, active_queries, avg_candidates = _multi_positive_sampled_softmax_loss(
                logits=logits,
                candidate_mask=batch["candidate_mask"],
                positive_mask=batch["candidate_positive_mask"],
            )

        if active_queries == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * active_queries
        total_batches += 1
        total_queries += active_queries
        total_avg_candidates += avg_candidates * active_queries
        if show_progress and total_queries > 0:
            iterator.set_postfix(
                loss=f"{total_loss / total_queries:.4f}",
                cand=f"{total_avg_candidates / total_queries:.1f}",
            )

    if total_queries == 0:
        return {
            "loss": 0.0,
            "batches": 0.0,
            "queries": 0.0,
            "avg_candidates": 0.0,
        }
    return {
        "loss": total_loss / total_queries,
        "batches": float(total_batches),
        "queries": float(total_queries),
        "avg_candidates": float(total_avg_candidates / total_queries),
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

    topk = parse_topk(args.topk)
    if not topk:
        raise ValueError("topk is empty.")

    print("[Recall] loading interactions ...")
    train_df = load_interactions(
        csv_path=train_path,
        max_rows=args.train_max_rows,
        sample_frac=1.0,
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
        raise RuntimeError("train_df is empty after filtering.")

    print(f"[Recall] train rows={len(train_df)}, valid rows={len(valid_df)}")
    print("[Recall] building feature stores ...")
    bucket_sizes = dict(DEFAULT_BUCKET_SIZES)
    user_store = UserFeatureStore.from_csv(user_path, bucket_sizes=bucket_sizes)
    train_candidate_ids = np.unique(train_df["video_id"].to_numpy(dtype=np.int64))
    train_item_store = ItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=bucket_sizes,
        candidate_video_ids=train_candidate_ids,
    )

    # Filter training rows whose item feature is absent.
    _, _, keep = train_item_store.lookup(train_df["video_id"].to_numpy(dtype=np.int64))
    train_df = train_df[keep].reset_index(drop=True)
    print(
        f"[Recall] train rows after item join={len(train_df)}, "
        f"train_items={train_item_store.num_items}"
    )
    if len(train_df) == 0:
        raise RuntimeError("No training rows left after item-feature join.")

    eval_item_store = ItemFeatureStore.from_csv(
        item_path,
        bucket_sizes=bucket_sizes,
        candidate_video_ids=None,
    )
    print(f"[Recall] eval candidate items={eval_item_store.num_items}")
    content_assets = None
    if args.content_recall_topn > 0:
        if args.content_item_emb is None or args.content_video_id_to_index is None:
            raise ValueError(
                "content recall is enabled but --content-item-emb/--content-video-id-to-index were not provided."
            )
        content_assets = load_content_recall_assets(
            embedding_path=args.content_item_emb,
            video_id_to_index_path=args.content_video_id_to_index,
            candidate_item_ids=eval_item_store.item_ids,
            device=device,
        )
        print(f"[Recall] content candidate items={len(content_assets.item_ids)}")
    hstu_assets = None
    if args.hstu_recall_topn > 0:
        if args.hstu_ckpt is None or args.hstu_data_dir is None:
            raise ValueError("HSTU recall is enabled but --hstu-ckpt/--hstu-data-dir were not provided.")
        hstu_assets = load_hstu_route_assets(
            ckpt_path=args.hstu_ckpt,
            data_dir=args.hstu_data_dir,
            candidate_video_ids=eval_item_store.item_ids,
            split_name="valid",
            device=device,
        )
        print(
            f"[Recall] hstu candidate items={len(hstu_assets.candidate_video_ids)} "
            f"eval rows={len(hstu_assets.eval_lookup)}"
        )

    item_counts = train_df["video_id"].value_counts(sort=False)
    sampler_item_ids = item_counts.index.to_numpy(dtype=np.int64, copy=False)
    sampler_item_counts = item_counts.to_numpy(dtype=np.float64, copy=False)
    hard_negative_sampler = RecallHardNegativeSampler(
        item_store=train_item_store,
        candidate_item_ids=sampler_item_ids,
        candidate_item_counts=sampler_item_counts,
    )

    train_ds = RecallTrainQueryDataset(
        frame=train_df,
        positive_label_mode=args.label_mode,
        sample_frac=args.train_sample_frac,
        seed=args.seed,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No recall training queries were built from train_df.")
    print(
        f"[Recall] train queries={len(train_ds)} "
        f"avg_pos={train_ds.avg_positive_items:.2f} "
        f"avg_explicit_neg={train_ds.avg_explicit_negative_items:.2f}"
    )

    collator = RecallQueryBatchCollator(
        user_store=user_store,
        item_store=train_item_store,
        bucket_sizes=bucket_sizes,
        max_history_len=args.max_history_len,
        device=torch.device("cpu"),
        hard_negative_sampler=hard_negative_sampler,
        num_explicit_negatives=args.num_explicit_negatives,
        num_hard_negatives=args.num_hard_negatives,
        num_random_negatives=args.num_random_negatives,
        max_positive_items=args.max_positive_items_per_query,
        seed=args.seed,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_ds,
        **loader_kwargs,
    )
    # 验证集按所有正样本曝光做点式评测，
    # 每个曝光点使用其发生前的历史，因此序列会在 split 内逐步增长。
    eval_samples = build_pointwise_eval_samples(
        interactions=valid_df,
        positive_label_mode=args.label_mode,
        max_samples=args.max_eval_users,
    )
    print(f"[Recall] eval points={len(eval_samples)}")
    model = TwoTowerRecallModel(
        bucket_sizes=bucket_sizes,
        embedding_dim=args.embedding_dim,
        tower_dim=args.tower_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history = []
    best_score = -1.0
    best_metrics = {}
    best_path = output_dir / "recall_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            temperature=args.temperature,
            device=device,
            amp_enabled=amp_enabled,
            show_progress=not args.disable_progress,
            epoch=epoch,
        )
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        valid_metrics = {}
        train_elapsed = time.time() - t0
        valid_elapsed = 0.0
        if should_eval:
            t1 = time.time()
            valid_metrics = evaluate_recall(
                model=model,
                user_store=user_store,
                item_store=eval_item_store,
                eval_samples=eval_samples,
                topk=topk,
                bucket_sizes=bucket_sizes,
                max_history_len=args.max_history_len,
                device=device,
                recall_topn=args.recall_topn,
                item_batch_size=args.eval_item_batch_size,
                query_batch_size=args.eval_query_batch_size,
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
            valid_elapsed = time.time() - t1
        elapsed = train_elapsed + valid_elapsed
        main_k = topk[-1]
        score = valid_metrics.get(f"hr@{main_k}", best_score)

        record = {
            "epoch": epoch,
            "elapsed_sec": elapsed,
            "train_elapsed_sec": train_elapsed,
            "valid_elapsed_sec": valid_elapsed,
            "train": train_stats,
            "valid": valid_metrics,
        }
        history.append(record)
        if should_eval:
            print(
                f"[Recall][Epoch {epoch}] "
                f"loss={train_stats['loss']:.5f} "
                f"hr@{main_k}={score:.5f} "
                f"ndcg@{main_k}={valid_metrics[f'ndcg@{main_k}']:.5f} "
                f"train={train_elapsed:.1f}s valid={valid_elapsed:.1f}s"
            )
        else:
            print(
                f"[Recall][Epoch {epoch}] "
                f"loss={train_stats['loss']:.5f} "
                f"train={train_elapsed:.1f}s valid=skipped"
            )

        if should_eval and score > best_score:
            best_score = score
            best_metrics = valid_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "bucket_sizes": bucket_sizes,
                    "config": vars(args),
                },
                best_path,
            )

    write_ids_to_npy(output_dir / "candidate_item_ids.npy", eval_item_store.item_ids)
    write_ids_to_npy(output_dir / "train_candidate_item_ids.npy", train_item_store.item_ids)

    summary = {
        "best_valid_score": best_score,
        "best_valid_metrics": best_metrics,
        "epochs": history,
        "train_rows": int(len(train_df)),
        "train_queries": int(len(train_ds)),
        "valid_rows": int(len(valid_df)),
        "num_train_items": int(train_item_store.num_items),
        "num_eval_items": int(eval_item_store.num_items),
        "avg_positive_items_per_query": float(train_ds.avg_positive_items),
        "avg_explicit_negative_items_per_query": float(train_ds.avg_explicit_negative_items),
    }
    save_json(output_dir / "recall_train_summary.json", summary)
    print(f"[Recall] best checkpoint: {best_path}")
    print(f"[Recall] summary saved: {output_dir / 'recall_train_summary.json'}")


if __name__ == "__main__":
    main()
