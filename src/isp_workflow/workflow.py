"""Coordinator: decode once, prepare once, compare candidates, review, export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Sequence

import numpy as np
import PIL
from PIL import Image

from . import __version__
from .agents import (ExportAgent, IntakeAgent, ISPAgent,
                     QualityAgent, QualityPolicy, TuningAgent, TuningPolicy)
from .models import RenderConfig


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class QualityGateError(ValueError):
    """All candidates failed automatic checks. Diagnostic artifacts are retained."""


@dataclass(frozen=True)
class RunResult:
    output_path: Path
    selected_candidate: str
    quality_report_path: Path


class Coordinator:
    """Agents can be replaced individually when extending this reference workflow."""

    def __init__(self, *, tuning_policy: TuningPolicy | None = None,
                 quality_policy: QualityPolicy | None = None,
                 intake: IntakeAgent | None = None, isp: ISPAgent | None = None,
                 exporter: ExportAgent | None = None) -> None:
        self.intake = intake or IntakeAgent()
        self.isp = isp or ISPAgent()
        self.tuning = TuningAgent(tuning_policy)
        self.quality = QualityAgent(quality_policy)
        self.exporter = exporter or ExportAgent()

    def run(self, input_path: Path | str, output_dir: Path | str, *,
            manifest_path: Path | str | None = None, workers: int = 2,
            save_intermediates: bool = False) -> RunResult:
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("workers must be an integer between 1 and 8")
        input_path, output_dir = Path(input_path), Path(output_dir)
        manifest_path = Path(manifest_path) if manifest_path is not None else None
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise ValueError("Output directory must be new or empty; existing runs are preserved")
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = output_dir / ".run.lock"
        try:
            # Exclusive creation reserves the directory before any artifact writes.
            run_lock = lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ValueError("Another run already reserved this output directory") from exc
        try:
            if any(path != lock_path for path in output_dir.iterdir()):
                raise ValueError("Output directory must be empty when reserved")
        except Exception:
            run_lock.close()
            lock_path.unlink()
            raise
        events: list[dict] = []

        def log(agent: str, action: str, **details: object) -> None:
            events.append({"agent": agent, "action": action,
                           "time_utc": datetime.now(timezone.utc).isoformat(), **details})

        log("coordinator", "started", input=str(input_path), workers=workers)
        try:
            frame = self.intake.run(input_path, manifest_path)
            log("intake", "decoded", representation=frame.representation)
            linear = self.isp.prepare(frame)
            log("isp", "prepared", shape=list(linear.shape))
            metadata = frame.describe()
            metadata["working_image"] = {"shape": list(linear.shape),
                                         "dtype": str(linear.dtype), "space": "linear_srgb"}
            metadata["runtime"] = {"workflow": __version__, "python": platform.python_version(),
                                   "numpy": np.__version__, "pillow": PIL.__version__}
            write_json(output_dir / "metadata.json", metadata)
            if save_intermediates:
                np.save(output_dir / "intermediate.npy", linear, allow_pickle=False)
            candidates = self.tuning.run(frame, linear)
            log("tuning", "proposed", candidates=[c.to_dict() for c in candidates])
            write_json(output_dir / "isp_config.json", {
                "candidates": [c.to_dict() for c in candidates],
                "quality": asdict(self.quality.policy), "selected": None,
            })
            preview_dir = output_dir / "previews"
            preview_dir.mkdir()

            def evaluate(item: tuple[int, RenderConfig]) -> dict:
                index, config = item
                rgb = self.isp.render(linear, config)
                review = self.quality.evaluate(rgb, config)
                # Index filenames prevent a policy's display names becoming paths.
                filename = f"candidate_{index:02d}.png"
                Image.fromarray(rgb).save(preview_dir / filename, format="PNG")
                review["preview"] = f"previews/{filename}"
                return review

            with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as pool:
                reviews = list(pool.map(evaluate, enumerate(candidates)))
            for review in reviews:
                log("quality", "evaluated", **review)
            selected = self.quality.select(reviews)
            report = {
                "status": "passed_automatic_checks" if selected is not None else "failed",
                "policy": asdict(self.quality.policy), "candidates": reviews,
                "selected": candidates[selected].name if selected is not None else None,
                "warnings": frame.warnings,
                "limitations": [
                    "Automatic selection minimizes display clipping among passing candidates; ties follow configuration order.",
                    "Noise, color accuracy and detail preservation require reference captures and human review.",
                    "Near-black fraction and encoded luma are diagnostics, not acceptance criteria.",
                ],
            }
            report_path = output_dir / "quality_report.json"
            write_json(report_path, report)
            if selected is None:
                raise QualityGateError("All candidates failed the clipping gate; inspect quality_report.json and previews")
            chosen_config = candidates[selected]
            # Reopen the selected lossless preview instead of holding all renders in RAM.
            with Image.open(preview_dir / f"candidate_{selected:02d}.png") as preview:
                chosen_rgb = np.array(preview)
            exported = self.exporter.run(chosen_rgb, output_dir / "output.bmp")
            log("export", "verified", **exported)
            write_json(output_dir / "isp_config.json", {
                "candidates": [c.to_dict() for c in candidates],
                "quality": asdict(self.quality.policy), "selected": chosen_config.to_dict(),
            })
            report["export"] = exported
            write_json(report_path, report)
            log("coordinator", "completed", selected=chosen_config.name)
            return RunResult(output_dir / "output.bmp", chosen_config.name, report_path)
        except Exception as exc:
            log("coordinator", "failed", error=type(exc).__name__, message=str(exc))
            write_json(output_dir / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            try:
                write_json(output_dir / "agent_log.json", events)
            finally:
                run_lock.close()
                lock_path.unlink()


def load_configuration(path: Path | None) -> tuple[Sequence[RenderConfig] | None, QualityPolicy]:
    if path is None:
        return None, QualityPolicy()
    def unique_keys(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = value
        return result

    data = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_keys)
    if not isinstance(data, dict) or set(data) - {"candidates", "quality"}:
        raise ValueError("Configuration must be an object with candidates and/or quality")
    candidates = data.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
            raise ValueError("candidates must be a list of between 1 and 12 configurations")
        if not all(isinstance(item, dict) for item in candidates):
            raise ValueError("Each candidate must be an object")
        candidates = [RenderConfig(**item) for item in candidates]
    quality = data.get("quality", {})
    if not isinstance(quality, dict):
        raise ValueError("quality must be an object")
    return candidates, QualityPolicy(**quality)
