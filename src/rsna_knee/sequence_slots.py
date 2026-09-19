"""Frozen issue #34 metadata slots and target-specific feature attention."""

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

SLOT_SCHEMA = "rsna-knee-slot-attention-v1"
SLOT_ORDER = (
    "axial_fluid_sensitive",
    "sagittal_fluid_sensitive",
    "coronal_fluid_sensitive",
    "sagittal_non_fluid_sensitive",
)


def resolve_sequence_slot(plane: str | None, fluid_sensitive: str | None) -> int:
    """Normalize raw indexed competition metadata; unknown values stay unmatched."""
    normalized_plane = (plane or "").strip().casefold()
    fluid = (fluid_sensitive or "").strip().casefold()
    if fluid in {"yes", "true", "1"}:
        return {"axial": 0, "sagittal": 1, "coronal": 2}.get(normalized_plane, -1)
    if fluid in {"no", "false", "0"} and normalized_plane == "sagittal":
        return 3
    return -1


def slot_metadata(targets: Sequence[str]) -> dict[str, object]:
    return {
        "kind": "target_slot_attention",
        "version": SLOT_SCHEMA,
        "slot_order": list(SLOT_ORDER),
        "definitions": {
            name: {"Anatomical_Plane": plane, "Fluid_Sensitive": fluid}
            for name, plane, fluid in zip(
                SLOT_ORDER,
                ("axial", "sagittal", "coronal", "sagittal"),
                (True, True, True, False),
                strict=True,
            )
        },
        "Fat_Suppression": "unused",
        "normalization": "strip/casefold; yes/true/1; no/false/0; unknown unmatched",
        "unmatched_series": "excluded from slot aggregation",
        "feature_pooling": "equal valid anchor mean; equal series mean within slot",
        "attention": "Linear(512,12); masked softmax over slots independently per target",
        "classifier": "shared existing Linear(512,12)",
        "zero_slot_policy": "training-label priors; excluded from optimizer updates",
        "target_order": list(targets),
    }


class TargetSpecificSlotAttention(nn.Module):
    def __init__(self, feature_count: int, target_count: int):
        super().__init__()
        self.projection = nn.Linear(feature_count, target_count)

    def forward(
        self,
        features: Tensor,
        series_mask: Tensor,
        slot_indices: Tensor,
        head: nn.Linear,
    ) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or series_mask.shape != features.shape[:2]:
            raise ValueError(
                "Slot attention requires [study, series, feature] and matching mask"
            )
        if slot_indices.shape != series_mask.shape or bool(
            ((slot_indices < -1) | (slot_indices >= 4)).any()
        ):
            raise ValueError("Slot indices must match series and be -1 or 0..3")
        membership = slot_indices.unsqueeze(-1) == torch.arange(
            4, device=features.device
        )
        membership = membership & series_mask.bool().unsqueeze(-1)
        counts = membership.sum(1)
        slots = torch.einsum("bsf,bsk->bkf", features, membership.to(features.dtype))
        slots = slots / counts.clamp_min(1).unsqueeze(-1)
        available = counts > 0
        scores = cast(Tensor, self.projection(slots)).float()
        scores = scores.masked_fill(~available.unsqueeze(-1), -torch.inf)
        # An empty study has no softmax domain; avoid NaNs before its prior fallback.
        scores = torch.where(available.any(1)[:, None, None], scores, 0)
        weights = scores.softmax(1).masked_fill(~available.unsqueeze(-1), 0)
        logits = (weights * head(slots)).sum(1)
        return logits, weights
