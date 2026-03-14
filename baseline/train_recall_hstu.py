#!/usr/bin/env python3
"""Train KuaiRand HSTU recall route."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data_utils import save_json, set_seed
from hstu_kuairand_data import create_kuairand_hstu_dataloader
from hstu_kuairand_model import KuaiRandHSTURecModel
from hstu_user_features import HSTUUserFeatureStore, resolve_hstu_user_features_csv


def parse_topk(s: str) -> list[int]:
    values = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not values:
        raise ValueError("topk must not be empty")
    return sorted(set(values))


def _metric_keys(topk: list[int]) -> list[str]:
    keys: list[str] = []
    for k in topk:
        keys.append(f"hr@{k}")
        keys.append(f"ndcg@{k}")
    return keys


def _rank_to_metrics(rank: int, topk: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in topk:
        out[f"hr@{k}"] = 1.0 if rank <= k else 0.0
        out[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    return out


def parse_args() -> argparse.Namespace:
    default_num_workers = max(1, min(8, (os.cpu_count() or 4) // 2))
    parser = argparse.ArgumentParser(description="Train KuaiRand HSTU recall route.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/hstu_interleaved_firsttoken_len100"),
    )
    parser.add_argument("--user-features-csv", type=Path, default=None)
    parser.add_argument(
        "--disable-user-static-features",
        action="store_true",
        help="Do not load or use user static profile features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/checkpoints/hstu_recall_pure"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--train-num-neg", type=int, default=100)
    parser.add_argument("--loss-temperature", type=float, default=0.05)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--item-embedding-dim", type=int, default=128)
    parser.add_argument("--signal-embedding-dim", type=int, default=16)
    parser.add_argument("--user-embedding-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--linear-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=32)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--attention-activation", type=str, choices=["silu", "softmax", "none"], default="silu")
    parser.add_argument("--max-uih-len", type=int, default=100)
    parser.add_argument("--max-sequence-len", type=int, default=202)
    parser.add_argument("--token-layout", type=str, choices=["joint", "interleaved"], default="interleaved")
    parser.add_argument("--user-feature-mode", type=str, choices=["per_token", "first_token"], default="first_token")
    parser.add_argument("--use-position-bias", action="store_true")
    parser.add_argument("--use-time-bias", action="store_true")
    parser.add_argument("--time-num-buckets", type=int, default=128)
    parser.add_argument("--time-log-base", type=float, default=0.301)
    parser.add_argument("--scale-by-sqrt-d", action="store_true")
    parser.add_argument("--l2-norm-embeddings", action="store_true")
    parser.add_argument("--eval-item-pool", type=str, choices=["train", "all"], default="all")
    parser.add_argument("--topk", type=str, default="20,50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def _sample_negative_matrix(
    pos_ids: torch.Tensor,
    item_pool: np.ndarray,
    num_neg: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    if pos_ids.numel() == 0:
        return torch.zeros((0, num_neg), dtype=torch.long, device=device)
    neg = np.array(
        [[int(item_pool[rng.randrange(len(item_pool))]) for _ in range(num_neg)] for _ in range(pos_ids.numel())],
        dtype=np.int64,
    )
    pos_np = pos_ids.detach().cpu().numpy().reshape(-1, 1)
    dup = neg == pos_np
    while dup.any():
        rr, cc = np.where(dup)
        for r, c in zip(rr.tolist(), cc.tolist()):
            neg[r, c] = int(item_pool[rng.randrange(len(item_pool))])
        dup = neg == pos_np
    return torch.from_numpy(neg).to(device=device, dtype=torch.long)


def train_one_epoch(
    model: KuaiRandHSTURecModel,
    optimizer: torch.optim.Optimizer,
    train_loader,
    train_item_pool: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    epoch_seed: int,
    show_progress: bool,
) -> float:
    model.train()
    rng = random.Random(epoch_seed)
    total_loss = 0.0
    steps = 0

    iterator = tqdm(
        train_loader,
        total=len(train_loader),
        desc="HSTU Train",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in iterator:
        pre_out, enc_out = model.encode_batch(batch=batch, device=device, causal=True)
        if enc_out.size(1) < 2:
            continue
        query_dense = enc_out[:, :-1, :]
        pos_dense = pre_out.next_item_ids
        valid_mask = pre_out.next_item_mask
        if valid_mask.sum().item() == 0:
            continue

        query = query_dense[valid_mask]
        pos_ids = pos_dense[valid_mask]
        neg_ids = _sample_negative_matrix(
            pos_ids=pos_ids,
            item_pool=train_item_pool,
            num_neg=args.train_num_neg,
            rng=rng,
            device=device,
        )
        pos_emb = model.get_item_representations(pos_ids)
        neg_emb = model.get_item_representations(neg_ids)

        if args.l2_norm_embeddings:
            query = F.normalize(query, dim=-1)
            pos_emb = F.normalize(pos_emb, dim=-1)
            neg_emb = F.normalize(neg_emb, dim=-1)

        pos_logits = torch.sum(query * pos_emb, dim=-1, keepdim=True)
        neg_logits = torch.einsum("bd,bkd->bk", query, neg_emb)
        logits = torch.cat([pos_logits, neg_logits], dim=1) / args.loss_temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1
        if show_progress and steps > 0:
            iterator.set_postfix(loss=f"{total_loss / steps:.4f}")
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_full_ranking(
    model: KuaiRandHSTURecModel,
    eval_loader,
    candidate_item_ids: np.ndarray,
    topk: list[int],
    device: torch.device,
    l2_norm_embeddings: bool,
    show_progress: bool,
) -> dict[str, float]:
    item_pool_list = candidate_item_ids.astype(np.int64).tolist()
    totals = {k: 0.0 for k in _metric_keys(topk)}
    used = 0
    skipped = 0

    model.eval()
    iterator = tqdm(
        eval_loader,
        total=len(eval_loader),
        desc="HSTU Eval",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in iterator:
        query_out = model.get_query_output(batch=batch, device=device, causal=True)
        seq_lengths = batch["seq_lengths"].tolist()
        dense_seq_item_ids = batch["dense_seq_item_ids"].tolist()
        for i in range(query_out.query_states.size(0)):
            if not bool(query_out.valid_mask[i].item()):
                skipped += 1
                continue
            target = int(query_out.target_item_ids[i].item())
            length = int(seq_lengths[i])
            seen = set(int(x) for x in dense_seq_item_ids[i][: max(length - 1, 0)] if int(x) > 0)
            seen.discard(target)
            candidates = [it for it in item_pool_list if it not in seen]
            if target not in candidates:
                candidates.append(target)
            cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_from_query(query_out.query_states[i], cand_tensor)
            if l2_norm_embeddings:
                query = F.normalize(query_out.query_states[i], dim=-1)
                item_repr = F.normalize(model.get_item_representations(cand_tensor), dim=-1)
                scores = torch.matmul(item_repr, query)
            scores_np = scores.detach().cpu().numpy()
            target_idx = candidates.index(target)
            target_score = float(scores_np[target_idx])
            rank = 1 + int(np.sum(scores_np > target_score))
            metrics = _rank_to_metrics(rank, topk)
            for key, value in metrics.items():
                totals[key] += value
            used += 1

    if used == 0:
        return {**{k: 0.0 for k in totals}, "num_samples": 0.0, "num_skipped": float(skipped)}
    avg = {k: v / used for k, v in totals.items()}
    avg["num_samples"] = float(used)
    avg["num_skipped"] = float(skipped)
    return avg


def main() -> None:
    args = parse_args()
    args.topk = parse_topk(args.topk)
    if args.loss_temperature <= 0:
        raise ValueError("loss_temperature must be > 0")
    if args.token_layout == "interleaved":
        min_required_len = 2 * (args.max_uih_len + 1)
    else:
        min_required_len = args.max_uih_len + 1
    if args.max_sequence_len < min_required_len:
        args.max_sequence_len = min_required_len

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    show_progress = not args.disable_progress
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (args.data_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    num_items = int(metadata["stats"]["num_items"])
    max_user_id = int(metadata["stats"]["max_user_id"])
    if args.disable_user_static_features:
        user_features_csv = None
        user_feature_store = None
    else:
        user_features_csv = resolve_hstu_user_features_csv(args.user_features_csv, args.data_dir)
        user_feature_store = HSTUUserFeatureStore.from_csv(user_features_csv)
    train_item_ids = np.load(args.data_dir / "train_hstu_item_ids.npy").astype(np.int64)
    all_item_ids = np.load(args.data_dir / "candidate_hstu_item_ids.npy").astype(np.int64)
    eval_item_pool = train_item_ids if args.eval_item_pool == "train" else all_item_ids

    train_loader = create_kuairand_hstu_dataloader(
        csv_path=args.data_dir / "train.csv",
        batch_size=args.batch_size,
        max_uih_len=args.max_uih_len,
        num_targets=1,
        ignore_last_n=0,
        shuffle=True,
        num_workers=args.num_workers,
        user_feature_store=user_feature_store,
    )
    valid_loader = create_kuairand_hstu_dataloader(
        csv_path=args.data_dir / "valid.csv",
        batch_size=args.eval_batch_size,
        max_uih_len=args.max_uih_len,
        num_targets=1,
        ignore_last_n=0,
        shuffle=False,
        num_workers=args.num_workers,
        user_feature_store=user_feature_store,
    )
    test_loader = create_kuairand_hstu_dataloader(
        csv_path=args.data_dir / "test.csv",
        batch_size=args.eval_batch_size,
        max_uih_len=args.max_uih_len,
        num_targets=1,
        ignore_last_n=0,
        shuffle=False,
        num_workers=args.num_workers,
        user_feature_store=user_feature_store,
    )

    model = KuaiRandHSTURecModel(
        num_items=num_items,
        num_user_embeddings=max_user_id + 1,
        embedding_dim=args.embedding_dim,
        item_embedding_dim=args.item_embedding_dim,
        signal_embedding_dim=args.signal_embedding_dim,
        user_embedding_dim=args.user_embedding_dim,
        max_sequence_len=args.max_sequence_len,
        token_layout=args.token_layout,
        user_feature_mode=args.user_feature_mode,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        linear_dim=args.linear_dim,
        attention_dim=args.attention_dim,
        dropout_rate=args.dropout_rate,
        attention_activation=args.attention_activation,
        use_position_bias=args.use_position_bias,
        use_time_bias=args.use_time_bias,
        time_num_buckets=args.time_num_buckets,
        time_log_base=args.time_log_base,
        scale_by_sqrt_d=args.scale_by_sqrt_d,
        user_static_bucket_sizes=(
            user_feature_store.bucket_sizes if user_feature_store is not None else None
        ),
        num_user_static_numeric=(
            user_feature_store.num_numeric_features if user_feature_store is not None else 0
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -math.inf
    best_metrics: Dict[str, float] = {}
    history: List[dict] = []
    best_path = output_dir / "hstu_recall_model.pt"

    print("[HSTU] train samples:", len(train_loader.dataset))
    print("[HSTU] valid samples:", len(valid_loader.dataset), "test samples:", len(test_loader.dataset))
    print("[HSTU] num_items:", num_items, "train_item_pool:", len(train_item_ids), "eval_item_pool:", len(eval_item_pool))
    print("[HSTU] device:", device)
    print("[HSTU] user static features:", user_features_csv if user_features_csv is not None else "disabled")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            train_item_pool=train_item_ids,
            args=args,
            device=device,
            epoch_seed=args.seed + epoch,
            show_progress=show_progress,
        )
        valid_metrics = evaluate_full_ranking(
            model=model,
            eval_loader=valid_loader,
            candidate_item_ids=eval_item_pool,
            topk=args.topk,
            device=device,
            l2_norm_embeddings=args.l2_norm_embeddings,
            show_progress=show_progress,
        )
        elapsed = time.time() - t0
        main_k = args.topk[-1]
        score = float(valid_metrics.get(f"ndcg@{main_k}", 0.0))
        history.append(
            {
                "epoch": epoch,
                "elapsed_sec": elapsed,
                "train_loss": train_loss,
                "valid": valid_metrics,
            }
        )
        print(
            f"[HSTU][Epoch {epoch}] "
            f"loss={train_loss:.5f} "
            f"hr@{main_k}={valid_metrics.get(f'hr@{main_k}', 0.0):.5f} "
            f"ndcg@{main_k}={score:.5f} "
            f"time={elapsed:.1f}s"
        )
        if score > best_score:
            best_score = score
            best_metrics = valid_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "metadata": metadata,
                    "user_static_bucket_sizes": (
                        user_feature_store.bucket_sizes if user_feature_store is not None else {}
                    ),
                    "user_static_num_features": (
                        user_feature_store.num_numeric_features if user_feature_store is not None else 0
                    ),
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_full_ranking(
        model=model,
        eval_loader=test_loader,
        candidate_item_ids=eval_item_pool,
        topk=args.topk,
        device=device,
        l2_norm_embeddings=args.l2_norm_embeddings,
        show_progress=show_progress,
    )

    summary = {
        "best_valid_score": best_score,
        "best_valid_metrics": best_metrics,
        "test_metrics": test_metrics,
        "epochs": history,
        "num_items": num_items,
        "train_item_pool_size": int(len(train_item_ids)),
        "eval_item_pool_size": int(len(eval_item_pool)),
        "user_features_csv": str(user_features_csv) if user_features_csv is not None else None,
    }
    save_json(output_dir / "hstu_recall_summary.json", summary)
    print(f"[HSTU] best checkpoint: {best_path}")
    print(f"[HSTU] summary saved: {output_dir / 'hstu_recall_summary.json'}")


if __name__ == "__main__":
    main()
