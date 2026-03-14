#!/usr/bin/env python3
"""Encode item content text with multilingual E5 and save normalized embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

def _patch_torch_pytree_compat() -> None:
    """
    `transformers` on some envs expects torch.utils._pytree.register_pytree_node,
    while older torch versions only expose _register_pytree_node.
    """
    try:
        import torch.utils._pytree as _torch_pytree
    except Exception:
        return

    if hasattr(_torch_pytree, "register_pytree_node"):
        return
    if not hasattr(_torch_pytree, "_register_pytree_node"):
        return

    def _compat_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        to_str_fn=None,
        maybe_from_str_fn=None,
        *,
        serialized_type_name=None,
        to_dumpable_context=None,
        from_dumpable_context=None,
        flatten_with_keys_fn=None,
    ):
        del to_str_fn, maybe_from_str_fn, serialized_type_name, flatten_with_keys_fn
        return _torch_pytree._register_pytree_node(
            cls,
            flatten_fn,
            unflatten_fn,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )

    _torch_pytree.register_pytree_node = _compat_register_pytree_node


_patch_torch_pytree_compat()

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "transformers is required for encoding content embeddings. "
        "Install it in the active environment before running this script."
    ) from exc


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
    return pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode item content text with multilingual-e5-base.")
    parser.add_argument(
        "--item-text-jsonl",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/content_assets/item_text.jsonl"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-base",
    )
    parser.add_argument(
        "--output-emb",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/content_assets/item_content_emb.npy"),
    )
    parser.add_argument(
        "--output-video-id-to-index",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure/content_assets/video_id_to_index.json"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    args.output_emb.parent.mkdir(parents=True, exist_ok=True)
    args.output_video_id_to_index.parent.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    video_ids: List[int] = []
    texts: List[str] = []
    with args.item_text_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            video_ids.append(int(record["video_id"]))
            texts.append(str(record["content_text"]))

    if not texts:
        raise RuntimeError("No item text found in item_text_jsonl.")

    print(f"[content_emb] loading model {args.model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    all_embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            end = min(len(texts), start + args.batch_size)
            encoded = tokenizer(
                texts[start:end],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32, copy=False))

    emb = np.concatenate(all_embeddings, axis=0)
    mapping = {str(video_id): idx for idx, video_id in enumerate(video_ids)}
    np.save(args.output_emb, emb)
    with args.output_video_id_to_index.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=True, indent=2)

    print(f"[content_emb] saved emb shape={tuple(emb.shape)} to {args.output_emb}")
    print(f"[content_emb] saved mapping size={len(mapping)} to {args.output_video_id_to_index}")


if __name__ == "__main__":
    main()
