"""Shared contracts. Pixel arrays stay in memory; reports contain descriptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class RawFrame:
    pixels: np.ndarray
    representation: Literal["bayer", "linear_srgb"]
    metadata: dict[str, Any] = field(default_factory=dict)
    black_level: float | list[float] = 0.0
    white_level: float | list[float] = 1.0
    wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_matrix: list[list[float]] | None = None
    cfa_pattern: str | None = None
    crop: tuple[int, int, int, int] | None = None  # x, y, width, height
    orientation: int = 1  # EXIF orientation; apply after demosaicing
    stages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "shape": list(self.pixels.shape),
            "dtype": str(self.pixels.dtype),
            "metadata": self.metadata,
            "black_level": self.black_level,
            "white_level": self.white_level,
            "wb_gains": list(self.wb_gains),
            "color_matrix": self.color_matrix,
            "cfa_pattern": self.cfa_pattern,
            "crop": self.crop,
            "orientation": self.orientation,
            "stages": self.stages,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class RenderConfig:
    name: str = "baseline"
    exposure_ev: float = 0.0
    contrast: float = 1.0
    denoise: float = 0.0
    sharpen: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Candidate name must be a non-empty string")
        limits = {"exposure_ev": (-5.0, 5.0), "contrast": (0.25, 2.0),
                  "denoise": (0.0, 1.0), "sharpen": (0.0, 2.0)}
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{key} must be finite and between {low} and {high}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
