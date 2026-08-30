"""Pinned DINOv2 ViT-S/14 encoder for offline 224-pixel inference."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DINOV2_VITS14_BACKBONE = "dinov2_vits14"
DINOV2_VITS14_FEATURE_COUNT = 384
_PATCH_SIZE = 14
_PRETRAINING_IMAGE_SIZE = 518


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            DINOV2_VITS14_FEATURE_COUNT,
            kernel_size=_PATCH_SIZE,
            stride=_PATCH_SIZE,
        )
        self.norm = nn.Identity()

    def forward(self, value: Tensor) -> Tensor:
        return cast(Tensor, self.norm(self.proj(value).flatten(2).transpose(1, 2)))


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dimension = DINOV2_VITS14_FEATURE_COUNT
        self.num_heads = 6
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.proj = nn.Linear(dimension, dimension)
        self.proj_drop = nn.Dropout(0)

    def forward(self, value: Tensor) -> Tensor:
        batch, token_count, dimension = value.shape
        qkv = self.qkv(value).reshape(
            batch, token_count, 3, self.num_heads, dimension // self.num_heads
        )
        query, key, item = torch.unbind(qkv, 2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            item.transpose(1, 2),
        )
        return cast(
            Tensor,
            self.proj_drop(
                self.proj(
                    attended.transpose(1, 2).reshape(batch, token_count, dimension)
                )
            ),
        )


class _LayerScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(DINOV2_VITS14_FEATURE_COUNT))

    def forward(self, value: Tensor) -> Tensor:
        return value * self.gamma


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dimension = DINOV2_VITS14_FEATURE_COUNT
        self.fc1 = nn.Linear(dimension, dimension * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dimension * 4, dimension)
        self.drop = nn.Dropout(0)

    def forward(self, value: Tensor) -> Tensor:
        return cast(Tensor, self.drop(self.fc2(self.drop(self.act(self.fc1(value))))))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dimension = DINOV2_VITS14_FEATURE_COUNT
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.attn = _Attention()
        self.ls1 = _LayerScale()
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dimension, eps=1e-6)
        self.mlp = _Mlp()
        self.ls2 = _LayerScale()
        self.drop_path2 = nn.Identity()

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.ls1(self.attn(self.norm1(value)))
        return cast(Tensor, value + self.ls2(self.mlp(self.norm2(value))))


class DinoV2ViTS14(nn.Module):
    """Meta DINOv2 ViT-S/14 without network loading or auxiliary token outputs."""

    def __init__(self) -> None:
        super().__init__()
        dimension = DINOV2_VITS14_FEATURE_COUNT
        patch_count = (_PRETRAINING_IMAGE_SIZE // _PATCH_SIZE) ** 2
        self.patch_embed = _PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count + 1, dimension))
        self.blocks = nn.ModuleList(_Block() for _ in range(12))
        self.norm = nn.LayerNorm(dimension, eps=1e-6)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, dimension))

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1:] != (3, 224, 224):
            raise ValueError(
                "dinov2_vits14 requires RGB inputs shaped [N, 3, 224, 224]"
            )
        tokens = self.patch_embed(value)
        tokens = torch.cat((self.cls_token.expand(value.shape[0], -1, -1), tokens), 1)
        tokens = tokens + self._position_encoding(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return cast(Tensor, self.head(self.norm(tokens)[:, 0]))

    def _position_encoding(self, tokens: Tensor) -> Tensor:
        patch_positions = self.pos_embed[:, 1:].float()
        grid = math.isqrt(patch_positions.shape[1])
        output_grid = 224 // _PATCH_SIZE
        scale = (output_grid + 0.1) / grid
        patch_positions = F.interpolate(
            patch_positions.reshape(1, grid, grid, tokens.shape[-1]).permute(
                0, 3, 1, 2
            ),
            mode="bicubic",
            scale_factor=(scale, scale),
        )
        if patch_positions.shape[-2:] != (output_grid, output_grid):
            raise RuntimeError(
                "DINOv2 positional interpolation produced the wrong grid"
            )
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
            1, output_grid * output_grid, tokens.shape[-1]
        )
        return torch.cat((self.pos_embed[:, :1], patch_positions), 1).to(tokens.dtype)


def dinov2_vits14() -> DinoV2ViTS14:
    """Construct the one supported DINOv2 candidate without pretrained I/O."""
    return DinoV2ViTS14()
