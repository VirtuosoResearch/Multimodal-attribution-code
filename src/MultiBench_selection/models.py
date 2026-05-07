"""Models for multimodal data selection experiments."""
from __future__ import annotations

from math import prod

import torch
import torch.nn as nn


class FlatEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], out_dim: int,
                 dropout: float) -> None:
        super().__init__()
        input_dim = int(prod(input_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        out_dim: int,
        *,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(f"SequenceEncoder expects (T,D), got {input_shape}")
        in_dim = int(input_shape[-1])
        self.proj = nn.Linear(in_dim, out_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=num_heads,
            dim_feedforward=out_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        token_present = x.abs().sum(dim=-1) > 0
        padding_mask = ~token_present
        padding_mask[:, 0] = False
        h = self.encoder(self.proj(x), src_key_padding_mask=padding_mask)
        weights = token_present.float().unsqueeze(-1)
        pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class ImageEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], out_dim: int) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"ImageEncoder expects (C,H,W), got {input_shape}")
        channels = int(input_shape[0])
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def build_modality_encoder(
    input_shape: tuple[int, ...],
    out_dim: int,
    *,
    dropout: float,
    num_heads: int = 4,
    num_layers: int = 1,
) -> nn.Module:
    if len(input_shape) == 3:
        return ImageEncoder(input_shape, out_dim)
    if len(input_shape) == 2:
        return SequenceEncoder(
            input_shape, out_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout)
    return FlatEncoder(input_shape, out_dim, dropout)


class ConcatSelectionMLP(nn.Module):
    """Dataset-aware concat model with per-sample missing auxiliary modality."""

    def __init__(
        self,
        main_shape: tuple[int, ...],
        aux_shape: tuple[int, ...],
        hidden_dim: int,
        num_classes: int,
        *,
        depth: int = 2,
        dropout: float = 0.1,
        include_mask_feature: bool = True,
        encoder_layers: int = 1,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.include_mask_feature = include_mask_feature
        self.main_encoder = build_modality_encoder(
            main_shape, hidden_dim, dropout=dropout,
            num_heads=num_heads, num_layers=encoder_layers)
        self.aux_encoder = build_modality_encoder(
            aux_shape, hidden_dim, dropout=dropout,
            num_heads=num_heads, num_layers=encoder_layers)
        input_dim = hidden_dim * 2 + (1 if include_mask_feature else 0)
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(max(depth, 1)):
            layers.extend([
                nn.Linear(current, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current = hidden_dim
        self.fusion = nn.Sequential(*layers)
        self.classifier = nn.Linear(current, num_classes)

    def forward(
        self,
        main: torch.Tensor,
        aux: torch.Tensor,
        aux_available: torch.Tensor,
    ) -> torch.Tensor:
        aux_mask = aux_available.float().view(-1, 1)
        main_h = self.main_encoder(main)
        aux_h = self.aux_encoder(aux) * aux_mask
        pieces = [main_h, aux_h]
        if self.include_mask_feature:
            pieces.append(aux_mask)
        return self.classifier(self.fusion(torch.cat(pieces, dim=1)))


class TokenTransformerSelectionModel(nn.Module):
    """Dataset-aware transformer over encoded main and auxiliary modality tokens."""

    def __init__(
        self,
        main_shape: tuple[int, ...],
        aux_shape: tuple[int, ...],
        num_classes: int,
        *,
        proj_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layers = max(1, num_layers // 2)
        self.main_encoder = build_modality_encoder(
            main_shape, proj_dim, dropout=dropout,
            num_heads=num_heads, num_layers=encoder_layers)
        self.aux_encoder = build_modality_encoder(
            aux_shape, proj_dim, dropout=dropout,
            num_heads=num_heads, num_layers=encoder_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(1, 3, proj_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        try:
            self.fusion = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False)
        except TypeError:
            self.fusion = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(proj_dim)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(
        self,
        main: torch.Tensor,
        aux: torch.Tensor,
        aux_available: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = main.shape[0]
        aux_mask = aux_available.float().view(-1, 1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        main_tok = self.main_encoder(main).unsqueeze(1)
        aux_tok = (self.aux_encoder(aux) * aux_mask).unsqueeze(1)
        tokens = torch.cat([cls, main_tok, aux_tok], dim=1) + self.type_embed
        padding_mask = torch.zeros(
            batch_size, 3, dtype=torch.bool, device=main.device)
        padding_mask[:, 2] = aux_available.view(-1) <= 0
        encoded = self.fusion(tokens, src_key_padding_mask=padding_mask)
        return self.classifier(self.norm(encoded[:, 0]))
