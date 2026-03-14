from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from hstu_user_features import HSTU_USER_CATEGORICAL_COLUMNS


def _import_hstu_encoder():
    hstu_dir = Path("/home/hfx/generative_rec/model")
    hstu_path = str(hstu_dir)
    if hstu_path not in sys.path:
        sys.path.insert(0, hstu_path)
    from hstu import HSTUEncoder  # type: ignore

    return HSTUEncoder


HSTUEncoder = _import_hstu_encoder()


@dataclass
class KuaiRandHSTUPreprocessorOutput:
    seq_lengths: torch.Tensor
    num_targets: torch.Tensor
    dense_seq_embeddings: torch.Tensor
    dense_timestamps: torch.Tensor
    valid_mask: torch.Tensor
    token_seq_lengths: torch.Tensor
    next_item_ids: torch.Tensor
    next_item_mask: torch.Tensor
    query_indices: torch.Tensor
    query_valid_mask: torch.Tensor


class KuaiRandHSTUInputPreprocessor(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_user_embeddings: int,
        embedding_dim: int = 128,
        item_embedding_dim: int = 128,
        signal_embedding_dim: int = 16,
        user_embedding_dim: int = 16,
        max_sequence_len: int = 202,
        token_layout: str = "interleaved",
        user_feature_mode: str = "first_token",
        num_signal_levels: int = 4,
        dropout_rate: float = 0.1,
        user_static_bucket_sizes: dict[str, int] | None = None,
        num_user_static_numeric: int = 0,
    ) -> None:
        super().__init__()
        if max_sequence_len <= 0:
            raise ValueError("max_sequence_len must be > 0")
        if token_layout not in {"joint", "interleaved"}:
            raise ValueError("token_layout must be one of {'joint', 'interleaved'}")
        if user_feature_mode not in {"per_token", "first_token"}:
            raise ValueError("user_feature_mode must be one of {'per_token', 'first_token'}")
        if num_user_embeddings <= 0:
            raise ValueError("num_user_embeddings must be > 0")

        self.max_sequence_len = int(max_sequence_len)
        self.embedding_dim = int(embedding_dim)
        self.token_layout = token_layout
        self.user_feature_mode = user_feature_mode
        self.user_static_bucket_sizes = dict(user_static_bucket_sizes or {})
        self.num_user_static_numeric = int(num_user_static_numeric)
        self.user_static_feature_names = [
            name for name in HSTU_USER_CATEGORICAL_COLUMNS if name in self.user_static_bucket_sizes
        ]

        self.item_embedding = nn.Embedding(num_items + 1, item_embedding_dim, padding_idx=0)
        self.signal_embedding = nn.Embedding(num_signal_levels + 1, signal_embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_user_embeddings, user_embedding_dim)
        self.user_static_embeddings = nn.ModuleList(
            [
                nn.Embedding(int(self.user_static_bucket_sizes[name]), user_embedding_dim)
                for name in self.user_static_feature_names
            ]
        )
        self.user_static_num_proj = (
            nn.Linear(self.num_user_static_numeric, user_embedding_dim)
            if self.num_user_static_numeric > 0
            else None
        )
        user_feature_parts = 1 + len(self.user_static_embeddings) + (1 if self.user_static_num_proj is not None else 0)
        if user_feature_parts > 1:
            user_hidden_dim = max(user_embedding_dim, user_feature_parts * user_embedding_dim)
            self.user_feature_fusion = nn.Sequential(
                nn.Linear(user_feature_parts * user_embedding_dim, user_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(user_hidden_dim),
                nn.Linear(user_hidden_dim, user_embedding_dim),
                nn.LayerNorm(user_embedding_dim),
            )
        else:
            self.user_feature_fusion = nn.Identity()

        fused_input_dim = item_embedding_dim + signal_embedding_dim + user_embedding_dim
        hidden_dim = max(self.embedding_dim, fused_input_dim)
        self.feature_fusion = nn.Sequential(
            nn.Linear(fused_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )
        self.position_embedding = nn.Embedding(self.max_sequence_len, self.embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        batch: dict,
        device: torch.device | None = None,
    ) -> KuaiRandHSTUPreprocessorOutput:
        if device is None:
            device = self.item_embedding.weight.device

        seq_lengths = batch["seq_lengths"].to(device=device, dtype=torch.long)
        num_targets = batch["num_targets"].to(device=device, dtype=torch.long)
        user_ids = batch["user_ids"].to(device=device, dtype=torch.long)
        dense_seq_item_ids = batch["dense_seq_item_ids"].to(device=device, dtype=torch.long)
        dense_seq_timestamps = batch["dense_seq_timestamps"].to(device=device, dtype=torch.long)
        dense_seq_signals = batch["dense_seq_signals"].to(device=device, dtype=torch.long)

        if self.token_layout == "joint":
            token_item_ids = dense_seq_item_ids
            token_signal_ids = dense_seq_signals
            token_timestamps = dense_seq_timestamps
            token_seq_lengths = seq_lengths
        else:
            bsz, inter_len = dense_seq_item_ids.shape
            token_len = inter_len * 2
            token_item_ids = torch.zeros((bsz, token_len), dtype=torch.long, device=device)
            token_signal_ids = torch.zeros((bsz, token_len), dtype=torch.long, device=device)
            token_timestamps = torch.zeros((bsz, token_len), dtype=torch.long, device=device)

            token_item_ids[:, 0::2] = dense_seq_item_ids
            token_signal_ids[:, 1::2] = dense_seq_signals
            token_timestamps[:, 0::2] = dense_seq_timestamps
            token_timestamps[:, 1::2] = dense_seq_timestamps
            token_seq_lengths = seq_lengths * 2

        max_len = token_item_ids.size(1)
        if max_len == 0:
            valid_mask = torch.zeros((seq_lengths.size(0), 0), dtype=torch.bool, device=device)
        else:
            arange = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
            valid_mask = arange < token_seq_lengths.unsqueeze(1)

        item_emb = self.item_embedding(token_item_ids)
        signal_emb = self.signal_embedding(token_signal_ids)
        user_parts = [self.user_embedding(user_ids.clamp(min=0, max=self.user_embedding.num_embeddings - 1))]
        if len(self.user_static_embeddings) > 0:
            raw_user_static_cat = batch.get("user_static_cat")
            if raw_user_static_cat is None:
                user_static_cat = torch.zeros(
                    (user_ids.size(0), len(self.user_static_embeddings)),
                    dtype=torch.long,
                    device=device,
                )
            else:
                user_static_cat = raw_user_static_cat.to(device=device, dtype=torch.long)
            for feature_idx, emb in enumerate(self.user_static_embeddings):
                user_parts.append(emb(user_static_cat[:, feature_idx].long()))
        if self.user_static_num_proj is not None:
            raw_user_static_num = batch.get("user_static_num")
            if raw_user_static_num is None:
                user_static_num = torch.zeros(
                    (user_ids.size(0), self.num_user_static_numeric),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                user_static_num = raw_user_static_num.to(device=device, dtype=torch.float32)
            user_parts.append(self.user_static_num_proj(user_static_num))
        if len(user_parts) == 1:
            user_emb_per_user = user_parts[0]
        else:
            user_emb_per_user = self.user_feature_fusion(torch.cat(user_parts, dim=-1))
        user_emb = user_emb_per_user.unsqueeze(1).expand(-1, max_len, -1)
        if self.user_feature_mode == "first_token":
            first_token_mask = torch.zeros((user_emb.size(0), max_len, 1), dtype=torch.float32, device=device)
            if max_len > 0:
                first_token_mask[:, 0, 0] = 1.0
            user_emb = user_emb * first_token_mask
        else:
            user_emb = user_emb * valid_mask.unsqueeze(-1).to(dtype=user_emb.dtype)

        fused = torch.cat([item_emb, signal_emb, user_emb], dim=-1)
        dense_seq_embeddings = self.feature_fusion(fused)

        positions = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
        positions = torch.clamp(positions, max=self.max_sequence_len - 1)
        pos_emb = self.position_embedding(positions)
        dense_seq_embeddings = self.dropout(dense_seq_embeddings + pos_emb)
        dense_seq_embeddings = dense_seq_embeddings * valid_mask.unsqueeze(-1).to(dtype=dense_seq_embeddings.dtype)

        if max_len <= 1:
            next_item_ids = torch.zeros((seq_lengths.size(0), 0), dtype=torch.long, device=device)
            next_item_mask = torch.zeros((seq_lengths.size(0), 0), dtype=torch.bool, device=device)
        else:
            next_item_ids = token_item_ids[:, 1:]
            next_item_mask = valid_mask[:, :-1] & valid_mask[:, 1:] & (next_item_ids > 0)

        if self.token_layout == "joint":
            query_indices = token_seq_lengths - num_targets - 1
        else:
            query_indices = token_seq_lengths - (2 * num_targets) - 1
        query_valid_mask = query_indices >= 0
        query_indices = torch.clamp(query_indices, min=0)

        return KuaiRandHSTUPreprocessorOutput(
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            dense_seq_embeddings=dense_seq_embeddings,
            dense_timestamps=token_timestamps,
            valid_mask=valid_mask,
            token_seq_lengths=token_seq_lengths,
            next_item_ids=next_item_ids,
            next_item_mask=next_item_mask,
            query_indices=query_indices,
            query_valid_mask=query_valid_mask,
        )


@dataclass
class KuaiRandHSTUQueryOutput:
    query_states: torch.Tensor
    target_item_ids: torch.Tensor
    valid_mask: torch.Tensor


class KuaiRandHSTURecModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_user_embeddings: int,
        embedding_dim: int = 128,
        item_embedding_dim: int = 128,
        signal_embedding_dim: int = 16,
        user_embedding_dim: int = 16,
        max_sequence_len: int = 202,
        token_layout: str = "interleaved",
        user_feature_mode: str = "first_token",
        num_layers: int = 2,
        num_heads: int = 2,
        linear_dim: int = 64,
        attention_dim: int = 32,
        dropout_rate: float = 0.1,
        attention_activation: str = "silu",
        use_position_bias: bool = True,
        use_time_bias: bool = True,
        time_num_buckets: int = 128,
        time_log_base: float = 0.301,
        scale_by_sqrt_d: bool = True,
        user_static_bucket_sizes: dict[str, int] | None = None,
        num_user_static_numeric: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.item_embedding_dim = int(item_embedding_dim)
        self.user_static_bucket_sizes = dict(user_static_bucket_sizes or {})
        self.num_user_static_numeric = int(num_user_static_numeric)

        self.preprocessor = KuaiRandHSTUInputPreprocessor(
            num_items=num_items,
            num_user_embeddings=num_user_embeddings,
            embedding_dim=embedding_dim,
            item_embedding_dim=item_embedding_dim,
            signal_embedding_dim=signal_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            max_sequence_len=max_sequence_len,
            token_layout=token_layout,
            user_feature_mode=user_feature_mode,
            dropout_rate=dropout_rate,
            user_static_bucket_sizes=self.user_static_bucket_sizes,
            num_user_static_numeric=self.num_user_static_numeric,
        )
        self.encoder = HSTUEncoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            linear_dim=linear_dim,
            attention_dim=attention_dim,
            max_seq_len=max_sequence_len,
            attention_activation=attention_activation,
            use_position_bias=use_position_bias,
            use_time_bias=use_time_bias,
            time_num_buckets=time_num_buckets,
            time_log_base=time_log_base,
            scale_by_sqrt_d=scale_by_sqrt_d,
            dropout_rate=dropout_rate,
        )
        if item_embedding_dim == embedding_dim:
            self.item_output_proj = nn.Identity()
        else:
            self.item_output_proj = nn.Linear(item_embedding_dim, embedding_dim, bias=False)
            nn.init.xavier_uniform_(self.item_output_proj.weight)

    def encode_batch(
        self,
        batch: dict,
        device: torch.device | None = None,
        causal: bool = True,
    ) -> tuple[KuaiRandHSTUPreprocessorOutput, torch.Tensor]:
        pre_out = self.preprocessor(batch=batch, device=device)
        enc_out = self.encoder(
            x=pre_out.dense_seq_embeddings,
            valid_mask=pre_out.valid_mask,
            timestamps=pre_out.dense_timestamps,
            causal=causal,
        )
        return pre_out, enc_out

    def get_query_output(
        self,
        batch: dict,
        device: torch.device | None = None,
        causal: bool = True,
    ) -> KuaiRandHSTUQueryOutput:
        if device is None:
            device = self.preprocessor.item_embedding.weight.device
        pre_out, enc_out = self.encode_batch(batch=batch, device=device, causal=causal)
        row_idx = torch.arange(pre_out.seq_lengths.size(0), device=device, dtype=torch.long)
        query_states = enc_out[row_idx, pre_out.query_indices]
        target_item_ids = batch["target_item_ids"].to(device=device, dtype=torch.long)[:, 0]
        return KuaiRandHSTUQueryOutput(
            query_states=query_states,
            target_item_ids=target_item_ids,
            valid_mask=pre_out.query_valid_mask,
        )

    def get_item_representations(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.to(dtype=torch.long)
        item_emb = self.preprocessor.item_embedding(item_ids)
        return self.item_output_proj(item_emb)

    def score_from_query(self, query_states: torch.Tensor, candidate_item_ids: torch.Tensor) -> torch.Tensor:
        item_repr = self.get_item_representations(candidate_item_ids)
        if query_states.dim() == 1 and item_repr.dim() == 2:
            return torch.matmul(item_repr, query_states)
        if query_states.dim() == 2 and item_repr.dim() == 3:
            return torch.einsum("bd,bcd->bc", query_states, item_repr)
        raise ValueError(
            f"Unsupported shapes: query={tuple(query_states.shape)}, candidates={tuple(candidate_item_ids.shape)}"
        )
