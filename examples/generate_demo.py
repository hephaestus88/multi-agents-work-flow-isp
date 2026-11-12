"""Generate small, deterministic sensor fixtures; these are not real captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scene(width: int = 256, height: int = 192) -> np.ndarray:
    """Linear RGB color patches over a smooth neutral ramp."""
    ramp = np.linspace(0.015, 0.7, width, dtype=np.float32)
    rgb = np.broadcast_to(ramp[None, :, None], (height, width, 3)).copy()
    colors = np.array([
        [0.65, 0.06, 0.04], [0.04, 0.65, 0.06], [0.04, 0.06, 0.65],
        [0.55, 0.5, 0.04], [0.05, 0.5, 0.55], [0.5, 0.05, 0.5],
        [0.18, 0.18, 0.18], [0.6, 0.6, 0.6],
    ], dtype=np.float32)
    for index, color in enumerate(colors):
        left = index * width // len(colors)
        right = (index + 1) * width // len(colors)
        rgb[height // 4:3 * height // 4, left:right] = color
    return rgb


def mosaic(rgb: np.ndarray, black: int, white: int) -> np.ndarray:
    raw = np.empty(rgb.shape[:2], dtype=np.float32)
    for y, x, channel in ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)):
        raw[y::2, x::2] = rgb[y::2, x::2, channel]
    return np.rint(raw * (white - black) + black).astype(np.uint16)


def pack_raw10(samples: np.ndarray) -> bytes:
    groups = samples.reshape(samples.shape[0], -1, 4)
    packed = np.empty((*groups.shape[:2], 5), dtype=np.uint8)
    packed[..., :4] = groups >> 2
    packed[..., 4] = ((groups[..., 0] & 3) | ((groups[..., 1] & 3) << 2)
                       | ((groups[..., 2] & 3) << 4) | ((groups[..., 3] & 3) << 6))
    return packed.tobytes()


def pack_raw12(samples: np.ndarray) -> bytes:
    groups = samples.reshape(samples.shape[0], -1, 2)
    packed = np.empty((*groups.shape[:2], 3), dtype=np.uint8)
    packed[..., :2] = groups >> 4
    packed[..., 2] = (groups[..., 0] & 15) | ((groups[..., 1] & 15) << 4)
    return packed.tobytes()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_dng(path: Path, samples: np.ndarray, black: int, white: int) -> None:
    try:
        import tifffile
    except ImportError as exc:
        raise SystemExit("DNG fixture generation needs: pip install '.[dev]'") from exc
    # DNG ColorMatrix maps XYZ to camera coordinates. This synthetic camera's
    # channels are linear sRGB; use the D65 XYZ -> linear sRGB matrix.
    matrix = (3.2404542, -1.5371385, -0.4985314,
              -0.9692660, 1.8760108, 0.0415560,
              0.0556434, -0.2040259, 1.0572252)
    rational_matrix = tuple(item for value in matrix for item in
                            (int(round(value * 1_000_000)), 1_000_000))
    tags = [
        (274, "H", 1, 1, False),  # Orientation
        (33421, "H", 2, (2, 2), False),  # CFARepeatPatternDim
        (33422, "B", 4, (0, 1, 1, 2), False),  # RGGB
        (50706, "B", 4, (1, 4, 0, 0), False),  # DNGVersion
        (50707, "B", 4, (1, 3, 0, 0), False),  # DNGBackwardVersion
        (50708, "s", 0, "ISP workflow synthetic sensor", False),
        (50710, "B", 3, (0, 1, 2), False),  # CFAPlaneColor
        (50711, "H", 1, 1, False),  # CFALayout
        (50713, "H", 2, (1, 1), False),  # BlackLevelRepeatDim
        (50714, "I", 1, black, False),
        (50717, "I", 1, white, False),
        (50721, "2i", 9, rational_matrix, False),
        (50728, "2I", 3, (1, 1, 1, 1, 1, 1), False),  # AsShotNeutral
        (50778, "H", 1, 21, False),  # CalibrationIlluminant1: D65
        (50829, "I", 4, (0, 0, *samples.shape), False),  # ActiveArea
    ]
    tifffile.imwrite(path, samples, photometric=32803, metadata=None,
                     compression=None, extratags=tags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo-data"))
    parser.add_argument("--dng", action="store_true", help="Also write an uncompressed Bayer DNG")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rgb = scene()
    height, width = rgb.shape[:2]
    write_json(args.output / "sensor_profile.json", {
        "cfa_pattern_at_frame_origin": "RGGB",
        "white_balance": [1.0, 1.0, 1.0],
        "camera_to_linear_srgb": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    })
    for depth, black, pack in ((10, 64, pack_raw10), (12, 256, pack_raw12)):
        white = (1 << depth) - 1
        stem = f"frame_raw{depth}"
        samples = mosaic(rgb, black, white)
        (args.output / f"{stem}.raw").write_bytes(pack(samples))
        write_json(args.output / f"{stem}.json", {
            "layout": f"mipi_raw{depth}",
            "width": width,
            "height": height,
            "row_stride_bytes": width * depth // 8,
            "data_offset_bytes": 0,
            "black_level": black,
            "white_level": white,
            "sensor_profile": "sensor_profile.json",
        })
    if args.dng:
        write_dng(args.output / "frame.dng", mosaic(rgb, 256, 4095), 256, 4095)
    print(f"Generated RAW10, RAW12{' and DNG' if args.dng else ''} fixtures in {args.output}")


if __name__ == "__main__":
    main()
