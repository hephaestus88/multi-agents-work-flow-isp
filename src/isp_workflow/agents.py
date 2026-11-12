"""Small role agents with explicit handoffs and replaceable tuning policies.

These are deterministic local agents. No network or language-model service is
required. An LLM integration can implement TuningPolicy and return validated
RenderConfig values while leaving decoding and pixel math in ordinary code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .adapters import load_frame
from .models import RawFrame, RenderConfig
from .processing import ISPProcessor


class IntakeAgent:
    def run(self, input_path: Path, manifest_path: Path | None) -> RawFrame:
        return load_frame(input_path, manifest_path)


class ISPAgent:
    def __init__(self, processor: ISPProcessor | None = None) -> None:
        self.processor = processor or ISPProcessor()

    def prepare(self, frame: RawFrame) -> np.ndarray:
        linear = self.processor.prepare(frame)
        linear.setflags(write=False)
        return linear

    def render(self, linear: np.ndarray, config: RenderConfig) -> np.ndarray:
        return self.processor.render(linear, config)


class TuningPolicy(Protocol):
    def propose(self, frame: RawFrame, linear: np.ndarray) -> Sequence[RenderConfig]:
        """Return a bounded, ordered list of candidates; first wins exact ties."""
        ...


class PresetTuningPolicy:
    """One bounded comparison round; no unbounded optimization loop."""

    def __init__(self, candidates: Sequence[RenderConfig] | None = None) -> None:
        self.candidates = list(candidates) if candidates is not None else [
            RenderConfig(),
            RenderConfig(name="highlight_priority", exposure_ev=-0.5),
            RenderConfig(name="soft_detail", denoise=0.25, sharpen=0.2),
        ]

    def propose(self, frame: RawFrame, linear: np.ndarray) -> Sequence[RenderConfig]:
        return list(self.candidates)


class TuningAgent:
    def __init__(self, policy: TuningPolicy | None = None) -> None:
        self.policy = policy or PresetTuningPolicy()

    def run(self, frame: RawFrame, linear: np.ndarray) -> list[RenderConfig]:
        candidates = list(self.policy.propose(frame, linear))
        if not 1 <= len(candidates) <= 12:
            raise ValueError("Tuning policy must return between 1 and 12 candidates")
        if not all(isinstance(c, RenderConfig) for c in candidates):
            raise ValueError("Tuning policy must return RenderConfig objects")
        if len({c.name for c in candidates}) != len(candidates):
            raise ValueError("Candidate names must be unique")
        return candidates


@dataclass(frozen=True)
class QualityPolicy:
    max_clipped_fraction: float = 0.10

    def __post_init__(self) -> None:
        if (isinstance(self.max_clipped_fraction, bool)
                or not isinstance(self.max_clipped_fraction, (int, float))
                or not np.isfinite(self.max_clipped_fraction)
                or not 0 <= self.max_clipped_fraction <= 1):
            raise ValueError("max_clipped_fraction must be finite and between 0 and 1")


class QualityAgent:
    """Check structural validity and clipping; this does not certify image quality."""

    def __init__(self, policy: QualityPolicy | None = None) -> None:
        self.policy = policy or QualityPolicy()

    def evaluate(self, rgb: np.ndarray, config: RenderConfig) -> dict:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise ValueError("Quality review requires a non-empty uint8 RGB image")
        # These are display-code diagnostics, not calibrated scene measurements.
        clipped = float(np.mean(np.any(rgb == 255, axis=2)))
        dark = float(np.mean(np.all(rgb <= 1, axis=2)))
        mean_luma = float(np.mean(rgb @ np.array([0.2126, 0.7152, 0.0722])) / 255)
        return {
            "candidate": config.name,
            "passed": clipped <= self.policy.max_clipped_fraction,
            "clipped_pixel_fraction": clipped,
            "near_black_pixel_fraction": dark,
            "mean_encoded_luma": mean_luma,
            "dimensions": [int(rgb.shape[1]), int(rgb.shape[0])],
        }

    def select(self, reviews: Sequence[dict]) -> int | None:
        passed = [i for i, review in enumerate(reviews) if review["passed"]]
        if not passed:
            return None
        return min(passed, key=lambda i: (reviews[i]["clipped_pixel_fraction"], i))


class ExportAgent:
    def run(self, rgb: np.ndarray, output_path: Path) -> dict:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("BMP export requires uint8 RGB")
        Image.fromarray(rgb).save(output_path, format="BMP")
        with Image.open(output_path) as reopened:
            if reopened.format != "BMP" or reopened.mode != "RGB":
                raise ValueError("Exported file is not a 24-bit RGB BMP")
            if not np.array_equal(np.asarray(reopened), rgb):
                raise ValueError("BMP round-trip pixel verification failed")
        return {"file": output_path.name, "format": "BMP", "bits_per_pixel": 24,
                "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                "round_trip_verified": True}
