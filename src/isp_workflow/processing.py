"""Small, deterministic ISP reference implementation.

Bayer reconstruction uses normalized 3x3 bilinear interpolation. Denoising is a
box-filter blend and sharpening is an unsharp mask. These intentionally simple
algorithms are extension points, rather than a production camera tuning stack.
All calculations remain floating point until the final sRGB/BMP handoff.
"""

from __future__ import annotations

import numpy as np

from .models import RawFrame, RenderConfig

_BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}
_RAW_STAGES = {
    "normalization", "black_level_normalization", "white_balance", "demosaic",
    "color_conversion", "camera_to_linear_srgb",
}
_ENCODED_STAGES = {"gamma", "srgb_encoding", "quantize", "quantization", "bmp_export"}


def _numeric_array(value: object, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _site_values(value: object, label: str) -> np.ndarray:
    values = _numeric_array(value, label)
    if values.ndim == 0:
        return np.full((2, 2), values.item(), dtype=np.float32)
    if values.shape != (4,):
        raise ValueError(f"{label} must be a scalar or four CFA-site values in row-major order")
    return values.reshape(2, 2)


def _filter3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve spatial axes without modifying the input or mixing channels."""
    pad = ((1, 1), (1, 1)) + ((0, 0),) * (image.ndim - 2)
    # Reflect preserves the Bayer lattice phase; edge also supports 1-pixel RGB.
    mode = "reflect" if min(image.shape[:2]) > 1 else "edge"
    padded = np.pad(image, pad, mode=mode)
    height, width = image.shape[:2]
    result = np.zeros_like(image, dtype=np.float32)
    for row in range(3):
        for column in range(3):
            weight = kernel[row, column]
            if weight:
                result += weight * padded[row:row + height, column:column + width, ...]
    return result


def demosaic_bilinear(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    """Reconstruct RGB, preserving measured samples and constant-color borders."""
    if pattern not in _BAYER_PATTERNS:
        raise ValueError(f"Unsupported Bayer pattern: {pattern!r}")
    if mosaic.ndim != 2 or min(mosaic.shape) < 2:
        raise ValueError("Bayer data must be a 2D image at least 2 by 2 pixels")
    lattice = np.array(list(pattern)).reshape(2, 2)
    rows = np.arange(mosaic.shape[0]) % 2
    columns = np.arange(mosaic.shape[1]) % 2
    sites = lattice[rows[:, None], columns[None, :]]
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    channels = []
    for color in "RGB":
        mask = (sites == color).astype(np.float32)
        numerator = _filter3(mosaic * mask, kernel)
        denominator = _filter3(mask, kernel)
        channel = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        channels.append(np.where(mask != 0, mosaic, channel))
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def apply_orientation(rgb: np.ndarray, orientation: int) -> np.ndarray:
    """Apply the eight EXIF orientations after demosaicing and output cropping."""
    if type(orientation) is not int or orientation not in range(1, 9):
        raise ValueError("orientation must be an EXIF integer from 1 to 8")
    if orientation == 2:
        return rgb[:, ::-1]
    if orientation == 3:
        return rgb[::-1, ::-1]
    if orientation == 4:
        return rgb[::-1]
    if orientation == 5:
        return np.swapaxes(rgb, 0, 1)
    if orientation == 6:
        return np.rot90(rgb, -1, axes=(0, 1))
    if orientation == 7:
        return np.swapaxes(rgb, 0, 1)[::-1, ::-1]
    if orientation == 8:
        return np.rot90(rgb, 1, axes=(0, 1))
    return rgb


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Encode clipped linear sRGB with the piecewise sRGB transfer function."""
    values = np.clip(linear, 0.0, 1.0)
    return np.where(values <= 0.0031308, 12.92 * values,
                    1.055 * np.power(values, 1.0 / 2.4) - 0.055)


class ISPProcessor:
    """Stateless, thread-safe processing for independently rendered candidates."""

    def prepare(self, frame: RawFrame) -> np.ndarray:
        """Create camera-corrected linear sRGB once, preserving float headroom.

        A Bayer ``color_matrix`` maps camera RGB to linear sRGB; each output
        pixel is ``camera_rgb @ matrix.T``. Four black/white values describe
        the four positions of the CFA tile, rather than R/G/B/G channel order.
        The linear-sRGB adapter path has already completed these corrections.
        """
        pixels = _numeric_array(frame.pixels, "pixels")
        if frame.representation not in {"bayer", "linear_srgb"}:
            raise ValueError(f"Unsupported frame representation: {frame.representation!r}")
        stages = set(frame.stages)
        if stages & _ENCODED_STAGES:
            raise ValueError("Input frame is already display encoded or quantized")
        if frame.representation == "bayer":
            if stages & _RAW_STAGES:
                raise ValueError("Bayer frame contains already-applied ISP corrections")
            if pixels.ndim != 2 or min(pixels.shape) < 2:
                raise ValueError("Bayer data must be a 2D image at least 2 by 2 pixels")
            if frame.cfa_pattern not in _BAYER_PATTERNS:
                raise ValueError("cfa_pattern must be RGGB, BGGR, GRBG, or GBRG")
            black = _site_values(frame.black_level, "black_level")
            white = _site_values(frame.white_level, "white_level")
            if np.any(white <= black):
                raise ValueError("Every white level must exceed its corresponding black level")
            gains = _numeric_array(frame.wb_gains, "wb_gains")
            if gains.shape != (3,) or np.any(gains <= 0):
                raise ValueError("wb_gains must contain three positive RGB multipliers")
            rows = np.arange(pixels.shape[0]) % 2
            columns = np.arange(pixels.shape[1]) % 2
            black_map = black[rows[:, None], columns[None, :]]
            white_map = white[rows[:, None], columns[None, :]]
            normalized = (pixels - black_map) / (white_map - black_map)
            site_gains = np.array([gains["RGB".index(color)] for color in frame.cfa_pattern],
                                  dtype=np.float32).reshape(2, 2)
            normalized *= site_gains[rows[:, None], columns[None, :]]
            rgb = demosaic_bilinear(normalized, frame.cfa_pattern)
            if frame.color_matrix is not None:
                matrix = _numeric_array(frame.color_matrix, "color_matrix")
                if matrix.shape != (3, 3):
                    raise ValueError("color_matrix must be a 3 by 3 camera-to-linear-sRGB matrix")
                rgb = rgb @ matrix.T
        else:
            if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
                raise ValueError("Linear sRGB data must have shape height by width by 3")
            if frame.cfa_pattern is not None:
                raise ValueError("A linear_srgb frame must not carry an active CFA pattern")
            rgb = pixels.copy()

        if frame.crop is not None:
            if (len(frame.crop) != 4 or any(type(value) is not int for value in frame.crop)):
                raise ValueError("crop must contain integer x, y, width, height")
            x, y, width, height = frame.crop
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > rgb.shape[1] or y + height > rgb.shape[0]:
                raise ValueError("crop must be a positive rectangle within the decoded image")
            rgb = rgb[y:y + height, x:x + width]
        result = np.ascontiguousarray(apply_orientation(rgb, frame.orientation), dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("ISP preparation produced non-finite values; check levels and calibration")
        return result

    def render(self, linear: np.ndarray, config: RenderConfig) -> np.ndarray:
        """Render one candidate without mutating the shared prepared image."""
        if not isinstance(config, RenderConfig):
            raise ValueError("config must be a validated RenderConfig")
        rgb = _numeric_array(linear, "linear image")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("Linear image must have shape height by width by 3")
        try:
            with np.errstate(over="raise", invalid="raise"):
                rgb = rgb * np.float32(2.0 ** config.exposure_ev)
                rgb = (rgb - np.float32(0.18)) * np.float32(config.contrast) + np.float32(0.18)
                box = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
                if config.denoise:
                    rgb = rgb + np.float32(config.denoise) * (_filter3(rgb, box) - rgb)
                if config.sharpen:
                    rgb = rgb + np.float32(config.sharpen) * (rgb - _filter3(rgb, box))
        except FloatingPointError as exc:
            raise ValueError("Rendering overflowed; check image values and calibration") from exc
        return np.rint(srgb_encode(rgb) * 255.0).astype(np.uint8)
