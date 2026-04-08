"""Multimodal fusion models for Fisher geometry experiments.

Provides two model families:
  - ConcatFusionModel: simple concat + MLP baseline (original)
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
# Modern: Cross-Attention Transformer Fusion
# ---------------------------------------------------------------------------

class CrossFusionBlock(nn.Module):
    """Single cross-attention fusion block.

    X_out = FFN(LN(X + CrossAttn(Q=X, KV=Y)))
    Y_out = FFN(LN(Y + CrossAttn(Q=Y, KV=X)))
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

    def forward(self, x: torch.Tensor,
                y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x2 = self.norm_x1(x + self.cross_attn_x(x, y, y, need_weights=False)[0])
        y2 = self.norm_y1(y + self.cross_attn_y(y, x, x, need_weights=False)[0])
        x_out = self.norm_x2(x2 + self.ffn_x(x2))
        y_out = self.norm_y2(y2 + self.ffn_y(y2))
        return x_out, y_out


class CrossAttentionFusionModel(nn.Module):
    """Lightweight cross-attention fusion model representative of modern
    vision-language architectures (Flamingo, BLIP-2, PerceiverIO).

    Architecture:
        1. Per-modality linear projection to a shared dimension d.
        2. Learnable [CLS] token per modality.
        3. Stacked CrossFusionBlocks (bidirectional cross-attention + FFN).
        4. Classifier on pooled [CLS] tokens from both streams.

    When a modality is inactive its projected features are replaced with zeros
    so the cross-attention degenerates to self-attention on the active stream.
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
        self.pool_norm = nn.LayerNorm(proj_dim * 2)
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

    def forward(self, feat_x: torch.Tensor, feat_y: torch.Tensor,
                *, use_x: bool = True, use_y: bool = True) -> torch.Tensor:
        B = feat_x.shape[0]
        dev = feat_x.device
        hx = (self.proj_x(feat_x).unsqueeze(1) if use_x
              else torch.zeros(B, 1, self.proj_dim, device=dev))
        hy = (self.proj_y(feat_y).unsqueeze(1) if use_y
              else torch.zeros(B, 1, self.proj_dim, device=dev))
        seq_x = torch.cat([self.cls_x.expand(B, -1, -1), hx], dim=1)
        seq_y = torch.cat([self.cls_y.expand(B, -1, -1), hy], dim=1)
        for layer in self.layers:
            seq_x, seq_y = layer(seq_x, seq_y)
        pooled = torch.cat([seq_x[:, 0], seq_y[:, 0]], dim=1)
        return self.classifier(self.pool_norm(pooled))
