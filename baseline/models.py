#!/usr/bin/env python3
"""Model definitions for KuaiRand two-stage baseline."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import (
    CONTEXT_CATEGORICAL_COLUMNS,
    ITEM_CATEGORICAL_COLUMNS,
    ITEM_NUMERIC_COLUMNS,
    RANK_CONTEXT_CATEGORICAL_COLUMNS,
    RANK_CONTEXT_NUMERIC_COLUMNS,
    RANK_ITEM_CATEGORICAL_COLUMNS,
    RANK_ITEM_NUMERIC_COLUMNS,
    RANK_USER_CATEGORICAL_COLUMNS,
    RANK_USER_NUMERIC_COLUMNS,
    USER_CATEGORICAL_COLUMNS,
    USER_NUMERIC_COLUMNS,
)


def _build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    dims = [input_dim] + hidden_dims
    layers: List[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class TwoTowerRecallModel(nn.Module):
    """
        双塔召回
        物品塔综合物品字段
        用户塔综合上下文字段、用户字段和历史序列池化
        最终输出两个塔的输出向量
    """
    def __init__(
        self,
        bucket_sizes: Dict[str, int],
        embedding_dim: int = 16,
        tower_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bucket_sizes = dict(bucket_sizes)
        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.ModuleList(
            [nn.Embedding(bucket_sizes[col], embedding_dim) for col in USER_CATEGORICAL_COLUMNS]
        )
        self.item_embeddings = nn.ModuleList(
            [nn.Embedding(bucket_sizes[col], embedding_dim) for col in ITEM_CATEGORICAL_COLUMNS]
        )
        self.context_embeddings = nn.ModuleList(
            [nn.Embedding(bucket_sizes[col], embedding_dim) for col in CONTEXT_CATEGORICAL_COLUMNS]
        )

        self.user_num_proj = nn.Linear(len(USER_NUMERIC_COLUMNS), embedding_dim)
        self.item_num_proj = nn.Linear(len(ITEM_NUMERIC_COLUMNS), embedding_dim)

        user_in_dim = (
            len(USER_CATEGORICAL_COLUMNS) * embedding_dim
            + len(CONTEXT_CATEGORICAL_COLUMNS) * embedding_dim
            + embedding_dim
            + embedding_dim
        )
        item_in_dim = len(ITEM_CATEGORICAL_COLUMNS) * embedding_dim + embedding_dim

        self.user_mlp = _build_mlp(
            input_dim=user_in_dim,
            hidden_dims=[hidden_dim],
            output_dim=tower_dim,
            dropout=dropout,
        )
        self.item_mlp = _build_mlp(
            input_dim=item_in_dim,
            hidden_dims=[hidden_dim],
            output_dim=tower_dim,
            dropout=dropout,
        )

    def _pool_history(self, hist_ids: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        """
        用户历史视频序列池化为一个向量
        """
        # Use item_id embedding table for sequence pooling to align retrieval space.
        hist_emb = self.item_embeddings[0](hist_ids.long())
        mask = hist_mask.unsqueeze(-1).float()
        pooled = (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        return pooled

    def encode_user(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        ctx_cat: torch.Tensor,
        hist_ids: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        用户 多个离散字段embedidng拼接 [B,23d]
        上下文多个字段embedidng拼接 [B,3d]
        用户数值投影 [B,4]--->[B,d]
        历史池化[B,d]
        拼接后 经过user_mlp 得到用户塔输出 然后做L2norm
        """
        u_parts = [emb(user_cat[:, i].long()) for i, emb in enumerate(self.user_embeddings)]
        c_parts = [emb(ctx_cat[:, i].long()) for i, emb in enumerate(self.context_embeddings)]
        u_num = self.user_num_proj(user_num.float())
        h_pool = self._pool_history(hist_ids=hist_ids, hist_mask=hist_mask)

        x = torch.cat(u_parts + c_parts + [u_num, h_pool], dim=-1)
        x = self.user_mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def encode_item(self, item_cat: torch.Tensor, item_num: torch.Tensor) -> torch.Tensor:
        """
        物品侧同用户侧同理
        """
        i_parts = [emb(item_cat[:, i].long()) for i, emb in enumerate(self.item_embeddings)]
        i_num = self.item_num_proj(item_num.float())
        x = torch.cat(i_parts + [i_num], dim=-1)
        x = self.item_mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        ctx_cat: torch.Tensor,
        hist_ids: torch.Tensor,
        hist_mask: torch.Tensor,
        item_cat: torch.Tensor,
        item_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.encode_user(
            user_cat=user_cat,
            user_num=user_num,
            ctx_cat=ctx_cat,
            hist_ids=hist_ids,
            hist_mask=hist_mask,
        )
        i = self.encode_item(item_cat=item_cat, item_num=item_num)
        return u, i


class ContentTwoTowerRecallModel(nn.Module):
    """
    Learnable content recall with lightweight user-side signals.

    - user tower: lightweight context + weighted short-term content history +
      light long-term category preference
    - item tower: fixed content embedding -> trainable projection
    """

    def __init__(
        self,
        bucket_sizes: Dict[str, int],
        content_dim: int,
        embedding_dim: int = 16,
        tower_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        long_term_input_dim: int = 0,
        negative_immunity_mode: str = "off",
        negative_fixed_alpha: float = 0.5,
        negative_max_alpha: float = 1.0,
        negative_gate_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.bucket_sizes = dict(bucket_sizes)
        self.embedding_dim = int(embedding_dim)
        self.content_dim = int(content_dim)
        self.long_term_input_dim = int(long_term_input_dim)
        self.negative_immunity_mode = str(negative_immunity_mode)
        if self.negative_immunity_mode not in {"off", "fixed", "gated"}:
            raise ValueError(f"Unsupported negative_immunity_mode: {negative_immunity_mode}")
        self.negative_fixed_alpha = float(negative_fixed_alpha)
        self.negative_max_alpha = float(negative_max_alpha)

        self.context_feature_names = list(bucket_sizes.keys())
        self.context_embeddings = nn.ModuleList(
            [nn.Embedding(int(bucket_sizes[name]), embedding_dim) for name in self.context_feature_names]
        )
        self.short_history_proj = nn.Sequential(
            nn.Linear(self.content_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.long_term_proj = (
            nn.Sequential(
                nn.Linear(self.long_term_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            if self.long_term_input_dim > 0
            else None
        )
        self.negative_history_proj = (
            nn.Sequential(
                nn.Linear(self.content_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            if self.negative_immunity_mode != "off"
            else None
        )

        user_in_dim = (
            len(self.context_feature_names) * embedding_dim
            + embedding_dim
            + (embedding_dim if self.long_term_proj is not None else 0)
        )
        self.user_mlp = _build_mlp(
            input_dim=user_in_dim,
            hidden_dims=[hidden_dim],
            output_dim=tower_dim,
            dropout=dropout,
        )
        self.negative_query_mlp = (
            _build_mlp(
                input_dim=embedding_dim,
                hidden_dims=[hidden_dim],
                output_dim=tower_dim,
                dropout=dropout,
            )
            if self.negative_immunity_mode != "off"
            else None
        )
        gate_in_dim = (
            len(self.context_feature_names) * embedding_dim
            + embedding_dim
            + embedding_dim
            + (embedding_dim if self.long_term_proj is not None else 0)
            + 2
        )
        self.negative_gate_mlp = (
            nn.Sequential(
                nn.Linear(gate_in_dim, max(int(negative_gate_hidden_dim), 8)),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(max(int(negative_gate_hidden_dim), 8), 1),
            )
            if self.negative_immunity_mode == "gated"
            else None
        )
        self.item_mlp = _build_mlp(
            input_dim=self.content_dim,
            hidden_dims=[hidden_dim],
            output_dim=tower_dim,
            dropout=dropout,
        )

    def _pool_weighted_history(
        self,
        projector: nn.Module,
        hist_content_emb: torch.Tensor,
        hist_weights: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected = projector(hist_content_emb.float())
        weights = hist_weights.unsqueeze(-1).float() * hist_mask.unsqueeze(-1).float()
        pooled = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        return pooled

    def _pool_short_history(
        self,
        hist_content_emb: torch.Tensor,
        hist_weights: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._pool_weighted_history(
            projector=self.short_history_proj,
            hist_content_emb=hist_content_emb,
            hist_weights=hist_weights,
            hist_mask=hist_mask,
        )

    def _pool_negative_history(
        self,
        hist_content_emb: torch.Tensor,
        hist_weights: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.negative_history_proj is None:
            return torch.zeros(
                (hist_content_emb.size(0), self.embedding_dim),
                dtype=hist_content_emb.dtype,
                device=hist_content_emb.device,
            )
        return self._pool_weighted_history(
            projector=self.negative_history_proj,
            hist_content_emb=hist_content_emb,
            hist_weights=hist_weights,
            hist_mask=hist_mask,
        )

    def encode_user(
        self,
        light_ctx_cat: torch.Tensor,
        hist_content_emb: torch.Tensor,
        hist_weights: torch.Tensor,
        hist_mask: torch.Tensor,
        long_term_pref: torch.Tensor | None = None,
        neg_hist_content_emb: torch.Tensor | None = None,
        neg_hist_weights: torch.Tensor | None = None,
        neg_hist_mask: torch.Tensor | None = None,
        neg_history_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx_parts = [emb(light_ctx_cat[:, i].long()) for i, emb in enumerate(self.context_embeddings)]
        ctx_repr = torch.cat(ctx_parts, dim=-1)
        short_repr = self._pool_short_history(
            hist_content_emb=hist_content_emb,
            hist_weights=hist_weights,
            hist_mask=hist_mask,
        )
        pieces = ctx_parts + [short_repr]
        long_repr = None
        if self.long_term_proj is not None:
            if long_term_pref is None:
                long_term_pref = torch.zeros(
                    (light_ctx_cat.size(0), self.long_term_input_dim),
                    dtype=hist_content_emb.dtype,
                    device=hist_content_emb.device,
                )
            long_repr = self.long_term_proj(long_term_pref.float())
            pieces.append(long_repr)
        x = torch.cat(pieces, dim=-1)
        pos_query = self.user_mlp(x)
        if self.negative_immunity_mode == "off":
            return F.normalize(pos_query, p=2, dim=-1)

        if neg_hist_content_emb is None or neg_hist_weights is None or neg_hist_mask is None:
            neg_hist_content_emb = hist_content_emb.new_zeros((hist_content_emb.size(0), 0, self.content_dim))
            neg_hist_weights = hist_weights.new_zeros((hist_weights.size(0), 0))
            neg_hist_mask = hist_mask.new_zeros((hist_mask.size(0), 0))

        neg_repr = self._pool_negative_history(
            hist_content_emb=neg_hist_content_emb,
            hist_weights=neg_hist_weights,
            hist_mask=neg_hist_mask,
        )
        neg_query = self.negative_query_mlp(neg_repr)
        neg_active = (neg_hist_mask.sum(dim=1, keepdim=True) > 0).float()

        if neg_history_stats is None:
            neg_history_stats = hist_content_emb.new_zeros((hist_content_emb.size(0), 2))
        else:
            neg_history_stats = neg_history_stats.float()

        if self.negative_immunity_mode == "gated":
            gate_pieces = [ctx_repr, short_repr, neg_repr]
            if long_repr is not None:
                gate_pieces.append(long_repr)
            gate_pieces.append(neg_history_stats)
            gate_input = torch.cat(gate_pieces, dim=-1)
            gate = torch.sigmoid(self.negative_gate_mlp(gate_input)) * self.negative_max_alpha
        else:
            gate = pos_query.new_full((pos_query.size(0), 1), self.negative_fixed_alpha)
        gate = gate * neg_active
        final_query = pos_query - gate * neg_query
        return F.normalize(final_query, p=2, dim=-1)

    def encode_item(self, item_content_emb: torch.Tensor) -> torch.Tensor:
        x = self.item_mlp(item_content_emb.float())
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(
        self,
        light_ctx_cat: torch.Tensor,
        hist_content_emb: torch.Tensor,
        hist_weights: torch.Tensor,
        hist_mask: torch.Tensor,
        long_term_pref: torch.Tensor | None,
        item_content_emb: torch.Tensor,
        neg_hist_content_emb: torch.Tensor | None = None,
        neg_hist_weights: torch.Tensor | None = None,
        neg_hist_mask: torch.Tensor | None = None,
        neg_history_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.encode_user(
            light_ctx_cat=light_ctx_cat,
            hist_content_emb=hist_content_emb,
            hist_weights=hist_weights,
            hist_mask=hist_mask,
            long_term_pref=long_term_pref,
            neg_hist_content_emb=neg_hist_content_emb,
            neg_hist_weights=neg_hist_weights,
            neg_hist_mask=neg_hist_mask,
            neg_history_stats=neg_history_stats,
        )
        i = self.encode_item(item_content_emb=item_content_emb)
        return u, i


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim)) for _ in range(max(int(num_layers), 1))]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(max(int(num_layers), 1))]
        )
        for weight in self.weights:
            nn.init.normal_(weight, mean=0.0, std=0.02)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for weight, bias in zip(self.weights, self.biases):
            xw = torch.sum(xl * weight, dim=-1, keepdim=True)
            xl = x0 * xw + bias + xl
        return xl


class DINAttention(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mid_dim = max(hidden_dim // 2, 16)
        self.attn = _build_mlp(
            input_dim=embedding_dim * 4,
            hidden_dims=[hidden_dim, mid_dim],
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_expand = query.unsqueeze(1).expand_as(keys)
        attn_in = torch.cat(
            [query_expand, keys, query_expand - keys, query_expand * keys],
            dim=-1,
        )
        score = self.attn(attn_in).squeeze(-1)
        score = score.masked_fill(mask <= 0, -1e9)
        weight = torch.softmax(score, dim=-1)
        weight = weight * mask.float()
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(weight.unsqueeze(-1) * keys, dim=1)
        return pooled, weight


class DINDCNRanker(nn.Module):
    def __init__(
        self,
        bucket_sizes: Dict[str, int],
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dcn_layers: int = 2,
        dropout: float = 0.1,
        deep_hidden_dims: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.bucket_sizes = dict(bucket_sizes)
        self.embedding_dim = int(embedding_dim)
        self.deep_hidden_dims = list(deep_hidden_dims or [256, 128])

        self.user_embeddings = nn.ModuleList(
            [nn.Embedding(int(bucket_sizes[col]), self.embedding_dim) for col in RANK_USER_CATEGORICAL_COLUMNS]
        )
        self.item_embeddings = nn.ModuleList(
            [nn.Embedding(int(bucket_sizes[col]), self.embedding_dim) for col in RANK_ITEM_CATEGORICAL_COLUMNS]
        )
        self.context_embeddings = nn.ModuleList(
            [nn.Embedding(int(bucket_sizes[col]), self.embedding_dim) for col in RANK_CONTEXT_CATEGORICAL_COLUMNS]
        )

        self.user_num_proj = nn.Sequential(
            nn.Linear(len(RANK_USER_NUMERIC_COLUMNS), self.embedding_dim),
            nn.ReLU(),
        )
        self.item_num_proj = nn.Sequential(
            nn.Linear(len(RANK_ITEM_NUMERIC_COLUMNS), self.embedding_dim),
            nn.ReLU(),
        )
        self.ctx_num_proj = nn.Sequential(
            nn.Linear(len(RANK_CONTEXT_NUMERIC_COLUMNS), self.embedding_dim),
            nn.ReLU(),
        )
        self.din_attention = DINAttention(
            embedding_dim=self.embedding_dim,
            hidden_dim=max(int(hidden_dim), 32),
            dropout=dropout,
        )

        feature_dim = (
            len(RANK_USER_CATEGORICAL_COLUMNS) * self.embedding_dim
            + len(RANK_ITEM_CATEGORICAL_COLUMNS) * self.embedding_dim
            + len(RANK_CONTEXT_CATEGORICAL_COLUMNS) * self.embedding_dim
            + self.embedding_dim  # user dense
            + self.embedding_dim  # item numeric
            + self.embedding_dim  # context numeric
            + self.embedding_dim  # din pooled history
            + self.embedding_dim  # history author pooled
            + self.embedding_dim  # history tag pooled
            + self.embedding_dim  # target * history
            + self.embedding_dim  # target - history
        )
        self.cross_net = CrossNetwork(input_dim=feature_dim, num_layers=max(int(dcn_layers), 1))
        self.deep_net = _build_mlp(
            input_dim=feature_dim,
            hidden_dims=self.deep_hidden_dims,
            output_dim=self.deep_hidden_dims[-1],
            dropout=dropout,
        )
        head_in_dim = feature_dim + self.deep_hidden_dims[-1]
        self.output_layer = nn.Linear(head_in_dim, 1)

    def _mean_pool_history(self, hist_emb: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        mask = hist_mask.unsqueeze(-1).float()
        return (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

    def _encode_sparse_group(self, embeddings: nn.ModuleList, values: torch.Tensor) -> list[torch.Tensor]:
        return [emb(values[:, idx].long()) for idx, emb in enumerate(embeddings)]

    def forward(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        ctx_cat: torch.Tensor,
        ctx_num: torch.Tensor,
        hist_ids: torch.Tensor,
        hist_author_ids: torch.Tensor,
        hist_tag_ids: torch.Tensor,
        hist_mask: torch.Tensor,
        item_cat: torch.Tensor,
        item_num: torch.Tensor,
    ) -> torch.Tensor:
        user_parts = self._encode_sparse_group(self.user_embeddings, user_cat)
        item_parts = self._encode_sparse_group(self.item_embeddings, item_cat)
        ctx_parts = self._encode_sparse_group(self.context_embeddings, ctx_cat)

        user_num_repr = self.user_num_proj(user_num.float())
        item_num_repr = self.item_num_proj(item_num.float())
        ctx_num_repr = self.ctx_num_proj(ctx_num.float())

        target_video = item_parts[0]
        hist_emb = self.item_embeddings[0](hist_ids.long())
        # DIN keeps the query anchored on the current video embedding.
        hist_att, _ = self.din_attention(query=target_video, keys=hist_emb, mask=hist_mask)
        hist_author_emb = self.item_embeddings[1](hist_author_ids.long())
        hist_tag_emb = self.item_embeddings[7](hist_tag_ids.long())
        hist_author_pool = self._mean_pool_history(hist_emb=hist_author_emb, hist_mask=hist_mask)
        hist_tag_pool = self._mean_pool_history(hist_emb=hist_tag_emb, hist_mask=hist_mask)

        base = torch.cat(
            user_parts
            + item_parts
            + ctx_parts
            + [
                user_num_repr,
                item_num_repr,
                ctx_num_repr,
                hist_att,
                hist_author_pool,
                hist_tag_pool,
                target_video * hist_att,
                target_video - hist_att,
            ],
            dim=-1,
        )
        cross_out = self.cross_net(base)
        deep_out = self.deep_net(base)
        final = torch.cat([cross_out, deep_out], dim=-1)
        click_logit = self.output_layer(final).squeeze(-1)
        return click_logit


class SharedBottomRanker(nn.Module): 
    """
    SharedBottom多任务排序
    数据处理同召回部分
    """
    def __init__(
        self,
        bucket_sizes: Dict[str, int],
        embedding_dim: int = 16,
        shared_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bucket_sizes = dict(bucket_sizes)
        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.ModuleList(
            [nn.Embedding(bucket_sizes[col], embedding_dim) for col in USER_CATEGORICAL_COLUMNS]
        )
        self.item_embeddings = nn.ModuleList(
            [nn.Embedding(bucket_sizes[col], embedding_dim) for col in ITEM_CATEGORICAL_COLUMNS]
        )
        self.context_embeddings = nn.ModuleList(
            [nn.Embedding(bucket_sizes[col], embedding_dim) for col in CONTEXT_CATEGORICAL_COLUMNS]
        )

        self.user_num_proj = nn.Linear(len(USER_NUMERIC_COLUMNS), embedding_dim)
        self.item_num_proj = nn.Linear(len(ITEM_NUMERIC_COLUMNS), embedding_dim)

        user_repr_dim = len(USER_CATEGORICAL_COLUMNS) * embedding_dim + embedding_dim + embedding_dim
        item_repr_dim = len(ITEM_CATEGORICAL_COLUMNS) * embedding_dim + embedding_dim
        ctx_repr_dim = len(CONTEXT_CATEGORICAL_COLUMNS) * embedding_dim
        cross_dim = 1
        shared_input_dim = user_repr_dim + item_repr_dim + ctx_repr_dim + cross_dim

        self.shared_bottom = _build_mlp(
            input_dim=shared_input_dim,
            hidden_dims=[shared_dim],
            output_dim=shared_dim,
            dropout=dropout,
        )
        self.click_head = _build_mlp(
            input_dim=shared_dim,
            hidden_dims=[shared_dim // 2],
            output_dim=1,
            dropout=dropout,
        )
        self.long_view_head = _build_mlp(
            input_dim=shared_dim,
            hidden_dims=[shared_dim // 2],
            output_dim=1,
            dropout=dropout,
        )

    def _pool_history(self, hist_ids: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        hist_emb = self.item_embeddings[0](hist_ids.long())
        mask = hist_mask.unsqueeze(-1).float()
        pooled = (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        return pooled

    def encode_user(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        hist_ids: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        parts = [emb(user_cat[:, i].long()) for i, emb in enumerate(self.user_embeddings)]
        u_num = self.user_num_proj(user_num.float())
        h_pool = self._pool_history(hist_ids=hist_ids, hist_mask=hist_mask)
        return torch.cat(parts + [u_num, h_pool], dim=-1)

    def encode_item(self, item_cat: torch.Tensor, item_num: torch.Tensor) -> torch.Tensor:
        parts = [emb(item_cat[:, i].long()) for i, emb in enumerate(self.item_embeddings)]
        i_num = self.item_num_proj(item_num.float())
        return torch.cat(parts + [i_num], dim=-1)

    def encode_context(self, ctx_cat: torch.Tensor) -> torch.Tensor:
        parts = [emb(ctx_cat[:, i].long()) for i, emb in enumerate(self.context_embeddings)]
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        ctx_cat: torch.Tensor,
        hist_ids: torch.Tensor,
        hist_mask: torch.Tensor,
        item_cat: torch.Tensor,
        item_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        分为三个部分 用户表示 item表示和上下文表示
        同召回不同 上下文表示单独表示 不融合在用户特征中
        取用户和物品表示的后emb_dim维度相乘后再相加 最后concatenate
        输出两个任务头的logits
        """
        user_repr = self.encode_user(user_cat=user_cat, user_num=user_num, hist_ids=hist_ids, hist_mask=hist_mask)
        item_repr = self.encode_item(item_cat=item_cat, item_num=item_num)
        ctx_repr = self.encode_context(ctx_cat=ctx_cat)
        # A light explicit cross term helps stabilize early-stage training.
        cross = (user_repr[:, : self.embedding_dim] * item_repr[:, : self.embedding_dim]).sum(
            dim=-1, keepdim=True
        )

        shared_in = torch.cat([user_repr, item_repr, ctx_repr, cross], dim=-1)
        shared = self.shared_bottom(shared_in)
        click_logit = self.click_head(shared).squeeze(-1)
        long_view_logit = self.long_view_head(shared).squeeze(-1)
        return click_logit, long_view_logit
