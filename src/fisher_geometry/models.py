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
