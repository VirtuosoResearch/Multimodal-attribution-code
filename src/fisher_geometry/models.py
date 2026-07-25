"""Multimodal fusion models for Fisher geometry experiments.

Provides three model families:
  - ConcatFusionModel: simple concat + MLP baseline (original)
  - SharedMLPFusionModel: shared-weight MLP applied to each modality independently
  - CrossAttentionFusionModel: modern cross-attention transformer fusion
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Baseline: Concat-Fusion MLP
# ---------------------------------------------------------------------------

class ConcatFusionModel(nn.Module):
    """Single-hidden-layer MLP operating on concatenated modality features.

    f(x, y) = W2 * GELU(W1 [x; y] + b1) + b2
    When a modality is inactive its input is replaced with zeros.
    """

    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int,
                 num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fusion_in = nn.Linear(dim_x + dim_y, hidden_dim)
        self.fusion_hidden = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat_x: torch.Tensor, feat_y: torch.Tensor,
                *, use_x: bool = True, use_y: bool = True) -> torch.Tensor:
        if not use_x:
            feat_x = torch.zeros_like(feat_x)
        if not use_y:
            feat_y = torch.zeros_like(feat_y)
        h = torch.cat([feat_x, feat_y], dim=1)
        h = self.fusion_in(h)
        h = self.fusion_hidden(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Shared MLP Fusion
# ---------------------------------------------------------------------------

class SharedMLPFusionModel(nn.Module):
    """Shared-weight MLP applied to each modality independently.

    Each modality is first projected to a common dimension via per-modality
    linear layers, then processed through a shared MLP. The shared
    representations are summed and fed to a classifier.

    When a modality is inactive, only the active modality's representation
    is used (no zero-padding).
    """

    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int,
                 num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj_x = nn.Linear(dim_x, hidden_dim)
        self.proj_y = nn.Linear(dim_y, hidden_dim)
        self.shared_mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat_x: torch.Tensor, feat_y: torch.Tensor,
                *, use_x: bool = True, use_y: bool = True) -> torch.Tensor:
        h = torch.zeros(feat_x.shape[0], self.proj_x.out_features,
                        device=feat_x.device)
        count = 0
        if use_x:
            h = h + self.shared_mlp(self.proj_x(feat_x))
            count += 1
        if use_y:
            h = h + self.shared_mlp(self.proj_y(feat_y))
            count += 1
        if count > 1:
            h = h / count
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Modern: Cross-Attention Transformer Fusion
# ---------------------------------------------------------------------------

class CrossFusionBlock(nn.Module):
    """Single cross-attention fusion block.

    When both modalities are active:
        X_out = FFN(LN(X + CrossAttn(Q=X, KV=Y)))
        Y_out = FFN(LN(Y + CrossAttn(Q=Y, KV=X)))

    When one modality is inactive (use_x=False or use_y=False):
        Active stream uses self-attention instead of cross-attention.
        Inactive stream is skipped entirely.
    This ensures clean modality ablation with no gradient leakage through
    LayerNorm/bias on the inactive branch.
    """

    def __init__(self, dim: int, num_heads: int, ffn_mult: int,
                 dropout: float) -> None:
        super().__init__()
        self.cross_attn_x = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_x1 = nn.LayerNorm(dim)
        self.cross_attn_y = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_y1 = nn.LayerNorm(dim)
        self.ffn_x = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim), nn.Dropout(dropout))
        self.norm_x2 = nn.LayerNorm(dim)
        self.ffn_y = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim), nn.Dropout(dropout))
        self.norm_y2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                *, use_x: bool = True,
                use_y: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if use_x and use_y:
            # Both active: bidirectional cross-attention
            x2 = self.norm_x1(x + self.cross_attn_x(x, y, y, need_weights=False)[0])
            y2 = self.norm_y1(y + self.cross_attn_y(y, x, x, need_weights=False)[0])
            x_out = self.norm_x2(x2 + self.ffn_x(x2))
            y_out = self.norm_y2(y2 + self.ffn_y(y2))
        elif use_x and not use_y:
            # X active only: self-attention on X stream
            x2 = self.norm_x1(x + self.cross_attn_x(x, x, x, need_weights=False)[0])
            x_out = self.norm_x2(x2 + self.ffn_x(x2))
            y_out = y  # pass-through (will be ignored)
        elif use_y and not use_x:
            # Y active only: self-attention on Y stream
            y2 = self.norm_y1(y + self.cross_attn_y(y, y, y, need_weights=False)[0])
            y_out = self.norm_y2(y2 + self.ffn_y(y2))
            x_out = x  # pass-through (will be ignored)
        else:
            x_out, y_out = x, y
        return x_out, y_out


class CrossAttentionFusionModel(nn.Module):
    """Lightweight cross-attention fusion model representative of modern
    vision-language architectures (Flamingo, BLIP-2, PerceiverIO).

    Architecture:
        1. Per-modality linear projection to a shared dimension d.
        2. Learnable [CLS] token per modality.
        3. Stacked CrossFusionBlocks (bidirectional cross-attention + FFN).
        4. Classifier on pooled [CLS] tokens from both streams.

    When a modality is inactive, the active stream uses self-attention
    instead of cross-attention, and only the active stream's [CLS] token
    is used for classification (the inactive stream's half of the pooled
    vector is replaced with a learned null embedding). This ensures clean
    modality ablation: no gradient leakage through LayerNorm/bias on the
    inactive branch.
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        proj_dim: int = 256,
        num_heads: int = 4,
        ffn_mult: int = 4,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.proj_x = nn.Linear(dim_x, proj_dim)
        self.proj_y = nn.Linear(dim_y, proj_dim)
        self.cls_x = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.cls_y = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.layers = nn.ModuleList(
            [CrossFusionBlock(proj_dim, num_heads, ffn_mult, dropout)
             for _ in range(num_layers)])
        # Learned null embedding for the inactive stream's pooled slot
        self.null_embed = nn.Parameter(torch.zeros(1, proj_dim))
        self.pool_norm = nn.LayerNorm(proj_dim * 2)
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

    def forward(self, feat_x: torch.Tensor, feat_y: torch.Tensor,
                *, use_x: bool = True, use_y: bool = True) -> torch.Tensor:
        B = feat_x.shape[0]
        dev = feat_x.device

        # Project active modalities; inactive streams get a detached zero
        # (won't be used in attention — see CrossFusionBlock)
        if use_x:
            hx = self.proj_x(feat_x).unsqueeze(1)
            seq_x = torch.cat([self.cls_x.expand(B, -1, -1), hx], dim=1)
        else:
            seq_x = torch.zeros(B, 2, self.proj_dim, device=dev)

        if use_y:
            hy = self.proj_y(feat_y).unsqueeze(1)
            seq_y = torch.cat([self.cls_y.expand(B, -1, -1), hy], dim=1)
        else:
            seq_y = torch.zeros(B, 2, self.proj_dim, device=dev)

        for layer in self.layers:
            seq_x, seq_y = layer(seq_x, seq_y, use_x=use_x, use_y=use_y)

        # Pool: use active stream's CLS; replace inactive with null_embed
        pool_x = seq_x[:, 0] if use_x else self.null_embed.expand(B, -1)
        pool_y = seq_y[:, 0] if use_y else self.null_embed.expand(B, -1)
        pooled = torch.cat([pool_x, pool_y], dim=1)
        return self.classifier(self.pool_norm(pooled))


class TrafficPairGNNMLPFusionModel(nn.Module):
    """Traffic fusion model: tabular pair-GNN branch + vision MLP branch.

    Expects x modality in layout:
        [edge_tabular, node_u_tabular, node_v_tabular]
    where node_u/node_v chunks share the same dimension.

    The x branch first runs several message-passing style updates on the
    (u, v) node pair conditioned on edge features, then feeds pooled pair
    features into an MLP.

    The y branch mirrors the original traffic vision usage more explicitly:
    feat_y is split as [vision_u, vision_v], each endpoint is encoded with a
    shared vision MLP, and pair features are aggregated before fusion.
    """

    def __init__(
        self,
        *,
        dim_y: int,
        y_node_dim: int,
        x_node_dim: int,
        x_edge_dim: int,
        hidden_dim: int,
        num_classes: int,
        message_passing_layers: int = 2,
        classifier_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.y_node_dim = int(y_node_dim)
        self.expected_y_dim = int(2 * self.y_node_dim)
        self.x_node_dim = int(x_node_dim)
        self.x_edge_dim = int(x_edge_dim)
        self.expected_x_dim = int(self.x_edge_dim + 2 * self.x_node_dim)
        self.hidden_dim = int(hidden_dim)
        if int(dim_y) != self.expected_y_dim:
            raise ValueError(
                "TrafficPairGNNMLPFusionModel expected y dimension "
                f"{self.expected_y_dim}, got {int(dim_y)}. "
                "Please verify traffic_vision_node_dim."
            )

        self.node_proj = nn.Linear(self.x_node_dim, self.hidden_dim)
        self.edge_proj = (
            nn.Linear(self.x_edge_dim, self.hidden_dim)
            if self.x_edge_dim > 0
            else None
        )
        self.msg_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(max(int(message_passing_layers), 1))
            ]
        )
        self.msg_norm_u = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in self.msg_layers])
        self.msg_norm_v = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in self.msg_layers])

        self.x_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.y_node_head = nn.Sequential(
            nn.Linear(self.y_node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.y_pair_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.null_x = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.null_y = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.fusion_norm = nn.LayerNorm(self.hidden_dim * 2)

        layers: list[nn.Module] = []
        in_dim = self.hidden_dim * 2
        for _ in range(max(int(classifier_layers) - 1, 0)):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def _split_x(self, feat_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(feat_x.shape[1]) != self.expected_x_dim:
            raise ValueError(
                "TrafficPairGNNMLPFusionModel expected x dimension "
                f"{self.expected_x_dim}, got {int(feat_x.shape[1])}. "
                "Please verify traffic_edge_tabular_dim/traffic_node_tabular_dim."
            )
        edge_end = self.x_edge_dim
        node_u_end = edge_end + self.x_node_dim
        edge = feat_x[:, :edge_end] if self.x_edge_dim > 0 else feat_x.new_zeros(feat_x.shape[0], 0)
        node_u = feat_x[:, edge_end:node_u_end]
        node_v = feat_x[:, node_u_end:]
        return edge, node_u, node_v

    def _encode_x(self, feat_x: torch.Tensor) -> torch.Tensor:
        edge, node_u, node_v = self._split_x(feat_x)
        hu = self.node_proj(node_u)
        hv = self.node_proj(node_v)
        if self.edge_proj is not None:
            he = self.edge_proj(edge)
        else:
            he = hu.new_zeros(hu.shape[0], self.hidden_dim)

        for index, msg_layer in enumerate(self.msg_layers):
            msg_u = msg_layer(torch.cat([hu, hv, he], dim=1))
            msg_v = msg_layer(torch.cat([hv, hu, he], dim=1))
            hu = self.msg_norm_u[index](hu + msg_u)
            hv = self.msg_norm_v[index](hv + msg_v)

        pair_features = torch.cat([hu, hv, torch.abs(hu - hv), he], dim=1)
        return self.x_head(pair_features)

    def _encode_y(self, feat_y: torch.Tensor) -> torch.Tensor:
        if int(feat_y.shape[1]) != self.expected_y_dim:
            raise ValueError(
                "TrafficPairGNNMLPFusionModel expected y dimension "
                f"{self.expected_y_dim}, got {int(feat_y.shape[1])}. "
                "Please verify traffic_vision_node_dim."
            )
        vision_u = feat_y[:, : self.y_node_dim]
        vision_v = feat_y[:, self.y_node_dim :]
        hu = self.y_node_head(vision_u)
        hv = self.y_node_head(vision_v)
        pair = torch.cat([hu, hv, torch.abs(hu - hv)], dim=1)
        return self.y_pair_head(pair)

    def forward(
        self,
        feat_x: torch.Tensor,
        feat_y: torch.Tensor,
        *,
        use_x: bool = True,
        use_y: bool = True,
    ) -> torch.Tensor:
        batch_size = int(feat_x.shape[0])
        if use_x:
            x_repr = self._encode_x(feat_x)
        else:
            x_repr = self.null_x.expand(batch_size, -1)
        if use_y:
            y_repr = self._encode_y(feat_y)
        else:
            y_repr = self.null_y.expand(batch_size, -1)
        fused = torch.cat([x_repr, y_repr], dim=1)
        return self.classifier(self.fusion_norm(fused))
