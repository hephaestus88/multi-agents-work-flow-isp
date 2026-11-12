"""Command-line entry point."""

import argparse
from pathlib import Path
import sys

from .agents import PresetTuningPolicy
from .workflow import Coordinator, QualityGateError, load_configuration


def main(argv: list[str] | None = None) -> int:
    # A successful run must not fail while printing a Unicode path on Windows.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Agent-oriented DNG / RAW10 / RAW12 to BMP")
    parser.add_argument("input", type=Path, help="DNG or a single-frame sensor dump")
    parser.add_argument("--manifest", type=Path, help="Required capture JSON for sensor dumps")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty run directory")
    parser.add_argument("--config", type=Path, help="Candidate settings and quality policy JSON")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent candidates, 1 to 8")
    parser.add_argument("--save-intermediates", action="store_true", help="Save linear RGB float32 .npy")
    args = parser.parse_args(argv)
    try:
        candidates, quality = load_configuration(args.config)
        workflow = Coordinator(tuning_policy=PresetTuningPolicy(candidates), quality_policy=quality)
        result = workflow.run(args.input, args.output_dir, manifest_path=args.manifest,
                              workers=args.workers, save_intermediates=args.save_intermediates)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"isp-workflow: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, QualityGateError) else 1
    print(f"BMP: {result.output_path.resolve()}")
    print(f"Selected candidate: {result.selected_candidate}")
    print(f"Quality report: {result.quality_report_path.resolve()}")
    return 0
