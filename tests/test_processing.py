import numpy as np
import pytest

from isp_workflow.models import RawFrame, RenderConfig
from isp_workflow.processing import ISPProcessor, apply_orientation, demosaic_bilinear, srgb_encode


def mosaic_from_color(pattern, shape=(5, 7), rgb=(0.2, 0.4, 0.6)):
    lattice = np.array([rgb["RGB".index(c)] for c in pattern]).reshape(2, 2)
    return lattice[np.arange(shape[0])[:, None] % 2, np.arange(shape[1])[None, :] % 2].astype(np.float32)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("shape", [(2, 2), (5, 7), (6, 8)])
def test_bilinear_constant_color_including_boundaries(pattern, shape):
    actual = demosaic_bilinear(mosaic_from_color(pattern, shape), pattern)
    np.testing.assert_allclose(actual, np.broadcast_to([0.2, 0.4, 0.6], (*shape, 3)), atol=1e-6)


def test_measured_samples_are_preserved():
    mosaic = np.arange(35, dtype=np.float32).reshape(5, 7)
    actual = demosaic_bilinear(mosaic, "RGGB")
    np.testing.assert_array_equal(actual[0::2, 0::2, 0], mosaic[0::2, 0::2])
    np.testing.assert_array_equal(actual[1::2, 1::2, 2], mosaic[1::2, 1::2])
    np.testing.assert_array_equal(actual[0::2, 1::2, 1], mosaic[0::2, 1::2])
    np.testing.assert_array_equal(actual[1::2, 0::2, 1], mosaic[1::2, 0::2])


def test_prepare_site_levels_white_balance_and_matrix_direction():
    black = np.array([[10, 20], [30, 40]], dtype=np.float32)
    white = black + 100
    normalized = mosaic_from_color("RGGB")
    sites = (np.arange(5)[:, None] % 2, np.arange(7)[None, :] % 2)
    encoded = normalized * 100 + black[sites]
    matrix = [[1, 0.5, 0], [0, 1, 0], [0, 0, 2]]
    frame = RawFrame(encoded, "bayer", cfa_pattern="RGGB", black_level=black.ravel().tolist(),
                     white_level=white.ravel().tolist(), wb_gains=(2, 1, 0.5), color_matrix=matrix)
    actual = ISPProcessor().prepare(frame)
    np.testing.assert_allclose(actual, np.broadcast_to([0.6, 0.4, 0.6], (5, 7, 3)), atol=1e-6)


def test_headroom_and_negative_values_survive_preparation():
    frame = RawFrame(np.array([[0, 200], [200, 0]], dtype=np.uint16), "bayer",
                     cfa_pattern="RGGB", black_level=100, white_level=200)
    actual = ISPProcessor().prepare(frame)
    assert actual[0, 0, 0] == -1
    assert actual[0, 0, 1] == 1


def test_linear_srgb_bypasses_raw_corrections_and_preserves_input():
    pixels = np.full((2, 3, 3), 0.25, dtype=np.float32)
    frame = RawFrame(pixels, "linear_srgb", black_level=100, white_level=101,
                     wb_gains=(5, 5, 5), color_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                     stages=["normalization", "white_balance", "demosaic", "color_conversion"])
    actual = ISPProcessor().prepare(frame)
    np.testing.assert_array_equal(actual, pixels)
    actual[0, 0] = 0
    assert pixels[0, 0, 0] == 0.25


def test_odd_crop_after_demosaic_preserves_color_phase():
    frame = RawFrame(mosaic_from_color("RGGB"), "bayer", cfa_pattern="RGGB", crop=(1, 1, 3, 3))
    actual = ISPProcessor().prepare(frame)
    np.testing.assert_allclose(actual, np.broadcast_to([0.2, 0.4, 0.6], (3, 3, 3)), atol=1e-6)


@pytest.mark.parametrize("orientation,expected", [
    (1, [[1, 2, 3], [4, 5, 6]]), (2, [[3, 2, 1], [6, 5, 4]]),
    (3, [[6, 5, 4], [3, 2, 1]]), (4, [[4, 5, 6], [1, 2, 3]]),
    (5, [[1, 4], [2, 5], [3, 6]]), (6, [[4, 1], [5, 2], [6, 3]]),
    (7, [[6, 3], [5, 2], [4, 1]]), (8, [[3, 6], [2, 5], [1, 4]]),
])
def test_exif_orientations(orientation, expected):
    pixels = np.repeat(np.arange(1, 7).reshape(2, 3, 1), 3, axis=2)
    np.testing.assert_array_equal(apply_orientation(pixels, orientation)[..., 0], expected)


def test_piecewise_srgb_reference_values_and_quantization():
    samples = np.array([0, 0.0031308, 0.18, 1], dtype=np.float32)
    np.testing.assert_allclose(srgb_encode(samples), [0, 0.040449936, 0.46135613, 1], atol=1e-6)
    linear = np.broadcast_to(samples.reshape(1, 4, 1), (1, 4, 3))
    result = ISPProcessor().render(linear, RenderConfig())
    np.testing.assert_array_equal(result[0, :, 0], [0, 10, 118, 255])


def test_renderer_does_not_mutate_shared_float_image():
    linear = np.linspace(-0.2, 2, 5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
    before = linear.copy()
    linear.setflags(write=False)
    result = ISPProcessor().render(linear, RenderConfig(exposure_ev=1, denoise=0.3, sharpen=0.7))
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(linear, before)


def test_render_overflow_is_reported_instead_of_silently_cast_to_black():
    with pytest.raises(ValueError, match="overflowed"):
        ISPProcessor().render(np.full((2, 2, 3), 3e38, dtype=np.float32), RenderConfig(exposure_ev=2))


@pytest.mark.parametrize("kwargs", [
    {"black_level": [0, 0]}, {"black_level": 2, "white_level": 1},
    {"wb_gains": (1, 0, 1)}, {"color_matrix": [[1, 0], [0, 1]]},
    {"orientation": 9}, {"crop": (1, 1, 9, 9)}, {"cfa_pattern": "XXXX"},
    {"stages": ["demosaic"]}, {"stages": ["srgb_encoding"]},
])
def test_invalid_bayer_metadata_rejected(kwargs):
    parameters = {"cfa_pattern": "RGGB", **kwargs}
    with pytest.raises(ValueError):
        ISPProcessor().prepare(RawFrame(np.ones((4, 4)), "bayer", **parameters))


@pytest.mark.parametrize("pixels", [np.array([[np.nan]]), np.zeros((3, 4)), np.zeros((0, 2, 3))])
def test_invalid_linear_frame_rejected(pixels):
    with pytest.raises(ValueError):
        ISPProcessor().prepare(RawFrame(pixels, "linear_srgb"))


@pytest.mark.parametrize("kwargs", [
    {"exposure_ev": 6}, {"contrast": 0}, {"denoise": -0.1}, {"sharpen": 3}, {"exposure_ev": float("nan")},
])
def test_invalid_render_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        RenderConfig(**kwargs)
