"""Input adapters for explicit sensor layouts and LibRaw-supported DNG files.

The sensor adapter preserves integer codes for the shared ISP.  The DNG backend
owns camera calibration and returns *linear* sRGB; its completed stages are
recorded so downstream code cannot accidentally apply them a second time.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import RawFrame


_KEYS = {
    "input", "layout", "width", "height", "row_stride_bytes", "data_offset_bytes",
    "bit_depth", "byte_order", "bit_alignment", "cfa_pattern_at_frame_origin",
    "black_level", "white_level", "white_balance", "camera_to_linear_srgb",
    "crop", "orientation", "sensor_profile",
}
_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def _read_object(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    try:
        result = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Manifest/profile must be a JSON object: {path}")
    unknown = set(result) - _KEYS
    if unknown:
        raise ValueError(f"Unknown manifest/profile keys: {', '.join(sorted(unknown))}")
    return result, _sha256(data)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _levels(value: Any, name: str, maximum: int) -> float | list[float]:
    if isinstance(value, list):
        if len(value) != 4:
            raise ValueError(f"{name} must be a scalar or four CFA-site values in row-major order")
        result: float | list[float] = [_number(item, name) for item in value]
    else:
        result = _number(value, name)
    if np.any(np.asarray(result) < 0) or np.any(np.asarray(result) > maximum):
        raise ValueError(f"{name} must be within the sensor code range 0..{maximum}")
    return result


def unpack_raw10(rows: np.ndarray) -> np.ndarray:
    """Decode MIPI-style rows: four high bytes, then ascending low-bit pairs.

    ``rows`` has shape (height, payload_bytes_per_row), excluding row padding.
    """
    rows = np.asarray(rows)
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] % 5:
        raise ValueError("RAW10 input must be uint8 rows with a multiple of five payload bytes")
    groups = rows.reshape(rows.shape[0], -1, 5).astype(np.uint16)
    result = np.empty((*groups.shape[:2], 4), dtype=np.uint16)
    for pixel in range(4):
        result[..., pixel] = (groups[..., pixel] << 2) | ((groups[..., 4] >> (2 * pixel)) & 3)
    return result.reshape(rows.shape[0], -1)


def unpack_raw12(rows: np.ndarray) -> np.ndarray:
    """Decode MIPI-style rows: two high bytes followed by low nibbles."""
    rows = np.asarray(rows)
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] % 3:
        raise ValueError("RAW12 input must be uint8 rows with a multiple of three payload bytes")
    groups = rows.reshape(rows.shape[0], -1, 3).astype(np.uint16)
    result = np.empty((*groups.shape[:2], 2), dtype=np.uint16)
    result[..., 0] = (groups[..., 0] << 4) | (groups[..., 2] & 15)
    result[..., 1] = (groups[..., 1] << 4) | (groups[..., 2] >> 4)
    return result.reshape(rows.shape[0], -1)


def _load_sensor(input_path: Path, manifest_path: Path) -> RawFrame:
    capture, capture_hash = _read_object(manifest_path)
    resolved: dict[str, Any] = {}
    profile_hash: str | None = None
    profile_path: Path | None = None
    if "sensor_profile" in capture:
        if not isinstance(capture["sensor_profile"], str) or not capture["sensor_profile"]:
            raise ValueError("sensor_profile must name a JSON profile file")
        profile_path = (manifest_path.parent / capture["sensor_profile"]).resolve()
        resolved, profile_hash = _read_object(profile_path)
        if "sensor_profile" in resolved:
            raise ValueError("Nested sensor_profile references are unsupported")
    resolved.update(capture)
    required = {"layout", "width", "height", "row_stride_bytes", "cfa_pattern_at_frame_origin",
                "black_level", "white_level", "white_balance"}
    missing = required - resolved.keys()
    if missing:
        raise ValueError(f"Missing sensor metadata: {', '.join(sorted(missing))}")

    width = _integer(resolved["width"], "width", 2)
    height = _integer(resolved["height"], "height", 2)
    stride = _integer(resolved["row_stride_bytes"], "row_stride_bytes", 1)
    offset = _integer(resolved.get("data_offset_bytes", 0), "data_offset_bytes")
    layout = resolved["layout"]
    if not isinstance(layout, str) or layout not in {"mipi_raw10", "mipi_raw12", "unpacked16"}:
        raise ValueError("layout must be mipi_raw10, mipi_raw12, or unpacked16")
    if layout == "mipi_raw10":
        if width % 4:
            raise ValueError("mipi_raw10 width must be a multiple of four")
        bit_depth, payload_bytes = 10, width // 4 * 5
    elif layout == "mipi_raw12":
        if width % 2:
            raise ValueError("mipi_raw12 width must be a multiple of two")
        bit_depth, payload_bytes = 12, width // 2 * 3
    else:
        if "bit_depth" not in resolved:
            raise ValueError("unpacked16 requires bit_depth, byte_order, and bit_alignment")
        bit_depth = _integer(resolved["bit_depth"], "bit_depth", 1)
        if bit_depth not in (10, 12):
            raise ValueError("unpacked16 bit_depth must be 10 or 12")
        if resolved.get("byte_order") not in ("little", "big"):
            raise ValueError("unpacked16 byte_order must be little or big")
        if resolved.get("bit_alignment") not in ("lsb", "msb"):
            raise ValueError("unpacked16 bit_alignment must be lsb or msb")
        payload_bytes = width * 2
    if "bit_depth" in resolved:
        supplied_depth = _integer(resolved["bit_depth"], "bit_depth", 1)
        if supplied_depth != bit_depth:
            raise ValueError("bit_depth contradicts the selected layout")
    if layout != "unpacked16" and ({"byte_order", "bit_alignment"} & resolved.keys()):
        raise ValueError("byte_order and bit_alignment apply only to unpacked16")
    if stride < payload_bytes:
        raise ValueError(f"row_stride_bytes must be >= the {payload_bytes}-byte row payload")

    pattern = resolved["cfa_pattern_at_frame_origin"]
    if not isinstance(pattern, str) or pattern not in _PATTERNS:
        raise ValueError("cfa_pattern_at_frame_origin must be RGGB, BGGR, GRBG, or GBRG")
    maximum = (1 << bit_depth) - 1
    black = _levels(resolved["black_level"], "black_level", maximum)
    white = _levels(resolved["white_level"], "white_level", maximum)
    if np.any(np.asarray(white) <= np.asarray(black)):
        raise ValueError("Each white_level must exceed its corresponding black_level")
    wb = resolved["white_balance"]
    if not isinstance(wb, list) or len(wb) != 3:
        raise ValueError("white_balance must contain three positive RGB gains")
    wb_values = tuple(_number(item, "white_balance") for item in wb)
    if any(value <= 0 for value in wb_values):
        raise ValueError("white_balance gains must be positive")
    matrix = resolved.get("camera_to_linear_srgb")
    if matrix is not None:
        if (not isinstance(matrix, list) or len(matrix) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in matrix)):
            raise ValueError("camera_to_linear_srgb must be a finite 3x3 matrix")
        matrix = [[_number(value, "camera_to_linear_srgb") for value in row] for row in matrix]
        if np.linalg.matrix_rank(np.asarray(matrix)) < 3:
            raise ValueError("camera_to_linear_srgb must have full rank")
    crop = resolved.get("crop")
    if crop is not None:
        if not isinstance(crop, list) or len(crop) != 4:
            raise ValueError("crop must be [x, y, width, height]")
        crop = tuple(_integer(value, "crop") for value in crop)
        x, y, crop_width, crop_height = crop
        if crop_width < 2 or crop_height < 2 or x + crop_width > width or y + crop_height > height:
            raise ValueError("crop must describe an in-bounds area at least 2x2 pixels")
    orientation = _integer(resolved.get("orientation", 1), "orientation", 1)
    if orientation > 8:
        raise ValueError("orientation must be an EXIF orientation value from 1 to 8")

    data = input_path.read_bytes()
    expected_size = offset + height * stride
    if len(data) != expected_size:
        raise ValueError(f"Single-frame size mismatch: expected {expected_size} bytes, got {len(data)}")
    rows = np.frombuffer(data, dtype=np.uint8, offset=offset).reshape(height, stride)
    payload = np.ascontiguousarray(rows[:, :payload_bytes])
    if layout == "mipi_raw10":
        pixels = unpack_raw10(payload)
    elif layout == "mipi_raw12":
        pixels = unpack_raw12(payload)
    else:
        dtype = "<u2" if resolved["byte_order"] == "little" else ">u2"
        words = np.frombuffer(payload.tobytes(), dtype=dtype).reshape(height, width).astype(np.uint16)
        if resolved["bit_alignment"] == "msb":
            shift = 16 - bit_depth
            if np.any(words & ((1 << shift) - 1)):
                raise ValueError("Nonzero padding bits contradict unpacked16 msb alignment")
            pixels = words >> shift
        else:
            if np.any(words > maximum):
                raise ValueError("Sample exceeds bit_depth; check unpacked16 alignment/byte order")
            pixels = words

    warnings = []
    if matrix is None:
        warnings.append("No camera_to_linear_srgb calibration; identity conversion gives approximate color.")
    resolved["bit_depth"] = bit_depth
    return RawFrame(
        pixels=pixels, representation="bayer", black_level=black, white_level=white,
        wb_gains=wb_values, color_matrix=matrix, cfa_pattern=pattern, crop=crop,
        orientation=orientation, stages=["decode"], warnings=warnings,
        metadata={
            "source": str(input_path.resolve()), "source_sha256": _sha256(data),
            "manifest": str(manifest_path.resolve()), "manifest_sha256": capture_hash,
            "sensor_profile": str(profile_path) if profile_path else None,
            "sensor_profile_sha256": profile_hash, "resolved_manifest": resolved,
            "sample_units": "sensor_code_values",
        },
    )


def _load_dng(input_path: Path) -> RawFrame:
    try:
        rawpy = importlib.import_module("rawpy")
    except ImportError as exc:
        raise ValueError("DNG support requires rawpy: install with pip install '.[dng]'") from exc
    settings = {
        "gamma": (1.0, 1.0), "output_bps": 16, "use_camera_wb": True,
        "use_auto_wb": False, "no_auto_bright": True, "adjust_maximum_thr": 0.0,
        "bright": 1.0, "user_flip": 0,
    }
    warnings = [
        "DNG support follows the installed LibRaw backend; full DNG specification compliance is not guaranteed.",
        "DNG front-end uses bounded 16-bit linear output; highlight headroom clipped there cannot be recovered by later tuning.",
    ]
    try:
        with rawpy.imread(str(input_path)) as raw:
            flip = int(raw.sizes.flip)
            orientations = {0: 1, 3: 3, 5: 8, 6: 6}
            if flip not in orientations:
                raise ValueError(f"Unsupported LibRaw orientation code {flip}")
            source_wb = [float(value) for value in raw.camera_whitebalance]
            if len(source_wb) < 3 or any(not np.isfinite(value) or value <= 0 for value in source_wb[:3]):
                warnings.append("Camera white balance is unavailable or invalid; LibRaw may use its fallback white balance.")
            metadata = {
                "source": str(input_path.resolve()), "source_sha256": _file_sha256(input_path),
                "backend": "rawpy/LibRaw", "rawpy_version": str(rawpy.__version__),
                "libraw_version": ".".join(str(part) for part in rawpy.libraw_version)
                if isinstance(getattr(rawpy, "libraw_version", None), (tuple, list)) else None,
                "backend_settings": {**settings, "output_color": "sRGB"},
                "source_black_levels": [float(value) for value in raw.black_level_per_channel],
                "source_white_level": float(raw.white_level),
                "source_camera_white_balance": [value if np.isfinite(value) else None for value in source_wb],
                "source_orientation_code": flip,
                "sample_units": "normalized_linear_srgb",
            }
            rendered = raw.postprocess(output_color=rawpy.ColorSpace.sRGB, **settings)
            if rendered.dtype != np.uint16 or rendered.ndim != 3 or rendered.shape[2] != 3:
                raise ValueError("DNG backend returned an unsupported image representation")
            pixels = rendered.astype(np.float32) / np.float32(65535.0)
    except Exception as exc:
        raise ValueError(f"Cannot decode DNG with rawpy/LibRaw ({type(exc).__name__}): {exc}") from exc
    return RawFrame(
        pixels=pixels, representation="linear_srgb", metadata=metadata,
        orientation=orientations[flip],
        stages=["decode", "backend_normalization", "backend_white_balance", "backend_demosaic_if_needed",
                "backend_color_conversion"],
        warnings=warnings,
    )


def load_frame(input_path: Path, manifest_path: Path | None = None) -> RawFrame:
    """Decode one input. Bare sensor dumps require an explicit JSON manifest."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() == ".dng":
        if manifest_path is not None:
            raise ValueError("DNG reads embedded metadata; --manifest is for sensor dumps")
        return _load_dng(input_path)
    if manifest_path is None:
        raise ValueError("RAW10/RAW12 sensor dumps require --manifest with the exact capture layout")
    try:
        return _load_sensor(input_path, Path(manifest_path))
    except OSError as exc:
        raise ValueError(f"Cannot read sensor input/metadata: {exc}") from exc
