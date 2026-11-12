"""Decoder checks use published byte layouts and explicit expected samples."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from isp_workflow.adapters import load_frame, unpack_raw10, unpack_raw12


def manifest(tmp_path, **updates):
    config = {
        "layout": "mipi_raw10", "width": 4, "height": 2, "row_stride_bytes": 5,
        "cfa_pattern_at_frame_origin": "RGGB", "black_level": 0,
        "white_level": 1023, "white_balance": [1, 1, 1],
    }
    config.update(updates)
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_raw10_low_bit_order():
    # High bytes 00,55,aa,ff; low pairs 0,1,2,3 occupy bits [1:0]..[7:6].
    actual = unpack_raw10(np.array([[0x00, 0x55, 0xAA, 0xFF, 0xE4]], dtype=np.uint8))
    np.testing.assert_array_equal(actual, [[0, 341, 682, 1023]])


def test_raw12_low_nibble_order():
    actual = unpack_raw12(np.array([[0x12, 0xAB, 0xC3, 0x00, 0xFF, 0xF0]], dtype=np.uint8))
    np.testing.assert_array_equal(actual, [[0x123, 0xABC, 0, 4095]])


@pytest.mark.parametrize("decoder,width", [(unpack_raw10, 4), (unpack_raw12, 2)])
def test_unpack_rejects_incomplete_groups(decoder, width):
    with pytest.raises(ValueError):
        decoder(np.zeros((1, width), dtype=np.uint8))
    with pytest.raises(ValueError):
        decoder(np.zeros((1, 5 if width == 4 else 3), dtype=np.uint16))


def test_sensor_padding_header_and_provenance(tmp_path):
    source = tmp_path / "frame.raw"
    source.write_bytes(b"HEAD" + bytes([0, 0x55, 0xAA, 0xFF, 0xE4, 99, 99]) * 2)
    frame = load_frame(source, manifest(tmp_path, row_stride_bytes=7, data_offset_bytes=4))
    np.testing.assert_array_equal(frame.pixels, [[0, 341, 682, 1023]] * 2)
    assert frame.representation == "bayer"
    assert frame.cfa_pattern == "RGGB"
    assert len(frame.metadata["source_sha256"]) == 64
    assert len(frame.metadata["manifest_sha256"]) == 64
    assert "approximate color" in frame.warnings[0]


def test_sensor_raw12(tmp_path):
    source = tmp_path / "frame.raw12"
    source.write_bytes(bytes([0x12, 0xAB, 0xC3, 0, 0xFF, 0xF0]) * 2)
    frame = load_frame(source, manifest(tmp_path, layout="mipi_raw12", row_stride_bytes=6,
                                       white_level=4095))
    np.testing.assert_array_equal(frame.pixels, [[0x123, 0xABC, 0, 4095]] * 2)
    assert frame.metadata["resolved_manifest"]["bit_depth"] == 12


def test_profile_relative_path_and_capture_override(tmp_path):
    profile = {
        "black_level": [1, 2, 3, 4], "white_level": [1020, 1021, 1022, 1023],
        "white_balance": [2, 1, 1.5],
        "camera_to_linear_srgb": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    (tmp_path / "sensor.json").write_text(json.dumps(profile), encoding="utf-8")
    config = {"layout": "mipi_raw10", "width": 4, "height": 2, "row_stride_bytes": 5,
              "cfa_pattern_at_frame_origin": "BGGR", "sensor_profile": "sensor.json",
              "white_balance": [1.5, 1, 2]}
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    source = tmp_path / "frame.raw"
    source.write_bytes(bytes(10))
    frame = load_frame(source, path)
    assert frame.black_level == [1, 2, 3, 4]
    assert frame.wb_gains == (1.5, 1, 2)
    assert frame.metadata["sensor_profile_sha256"]
    assert frame.warnings == []


@pytest.mark.parametrize("byte_order", ["little", "big"])
@pytest.mark.parametrize("alignment", ["lsb", "msb"])
@pytest.mark.parametrize("depth", [10, 12])
def test_unpacked16_endian_and_alignment(tmp_path, byte_order, alignment, depth):
    expected = np.array([[0, 1, 17, (1 << depth) - 1]] * 2, dtype=np.uint16)
    words = expected << (16 - depth) if alignment == "msb" else expected
    words = words.astype("<u2" if byte_order == "little" else ">u2")
    source = tmp_path / "frame.raw"
    source.write_bytes(words.tobytes())
    path = manifest(tmp_path, layout="unpacked16", row_stride_bytes=8, bit_depth=depth,
                    white_level=(1 << depth) - 1, byte_order=byte_order, bit_alignment=alignment)
    np.testing.assert_array_equal(load_frame(source, path).pixels, expected)


@pytest.mark.parametrize("size", [9, 11])
def test_size_mismatch_rejects_truncation_and_extra_frames(tmp_path, size):
    source = tmp_path / "frame.raw"
    source.write_bytes(bytes(size))
    with pytest.raises(ValueError, match="size mismatch"):
        load_frame(source, manifest(tmp_path))


@pytest.mark.parametrize("updates,message", [
    ({"row_stride_bytes": 4}, "row_stride_bytes"),
    ({"width": 3}, "multiple of four"),
    ({"width": True}, "width"),
    ({"bit_depth": 12}, "contradicts"),
    ({"layout": []}, "layout must"),
    ({"cfa_pattern_at_frame_origin": "RGBG"}, "cfa_pattern"),
    ({"white_balance": [1, 0, 1]}, "positive"),
    ({"white_balance": [1, float("nan"), 1]}, "finite"),
    ({"black_level": 100, "white_level": 50}, "exceed"),
    ({"black_level": [0, 0]}, "four CFA-site"),
    ({"white_level": 4095}, "sensor code range"),
    ({"crop": [3, 0, 2, 2]}, "in-bounds"),
    ({"orientation": 9}, "orientation"),
    ({"byte_order": "little"}, "only to unpacked16"),
    ({"camera_to_linear_srgb": [[0, 0, 0]] * 3}, "full rank"),
    ({"sensor_profle": "typo.json"}, "Unknown"),
])
def test_bad_manifest_rejected(tmp_path, updates, message):
    source = tmp_path / "frame.raw"
    source.write_bytes(bytes(10))
    with pytest.raises(ValueError, match=message):
        load_frame(source, manifest(tmp_path, **updates))


def test_missing_calibration_is_explicit(tmp_path):
    path = manifest(tmp_path)
    config = json.loads(path.read_text())
    del config["black_level"]
    path.write_text(json.dumps(config))
    source = tmp_path / "frame.raw"
    source.write_bytes(bytes(10))
    with pytest.raises(ValueError, match="Missing sensor metadata: black_level"):
        load_frame(source, path)


@pytest.mark.parametrize("alignment,value", [("lsb", 4096), ("msb", 1)])
def test_unpacked_invalid_padding_or_range(tmp_path, alignment, value):
    source = tmp_path / "frame.raw"
    source.write_bytes(np.full((2, 4), value, dtype="<u2").tobytes())
    path = manifest(tmp_path, layout="unpacked16", bit_depth=12, row_stride_bytes=8,
                    byte_order="little", bit_alignment=alignment, white_level=4095)
    with pytest.raises(ValueError):
        load_frame(source, path)


def test_bare_dump_requires_manifest(tmp_path):
    source = tmp_path / "frame.raw"
    source.write_bytes(bytes(10))
    with pytest.raises(ValueError, match="require --manifest"):
        load_frame(source)


def test_dng_backend_contract_and_no_second_gamma(tmp_path, monkeypatch):
    source = tmp_path / "image.dng"
    source.write_bytes(b"synthetic backend-contract fixture")
    calls = []

    class FakeRaw:
        sizes = SimpleNamespace(flip=5)
        black_level_per_channel = [64] * 4
        white_level = 4095
        camera_whitebalance = [2, 1, 1.5, 1]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def postprocess(self, **kwargs):
            calls.append(kwargs)
            return np.full((4, 6, 3), 32768, dtype=np.uint16)

    fake = SimpleNamespace(imread=lambda _: FakeRaw(), __version__="test", libraw_version=(0, 22, 2),
                           ColorSpace=SimpleNamespace(sRGB="srgb"))
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    frame = load_frame(source)
    assert frame.representation == "linear_srgb"
    assert frame.cfa_pattern is None
    assert frame.orientation == 8
    assert frame.pixels.dtype == np.float32
    assert float(frame.pixels[0, 0, 0]) == pytest.approx(32768 / 65535)
    assert calls[0]["gamma"] == (1.0, 1.0)
    assert calls[0]["output_bps"] == 16
    assert calls[0]["user_flip"] == 0
    assert calls[0]["no_auto_bright"] is True
    assert frame.metadata["backend_settings"]["output_color"] == "sRGB"
    assert frame.metadata["libraw_version"] == "0.22.2"
    assert "backend_color_conversion" in frame.stages


def test_dng_missing_backend_has_actionable_error(tmp_path, monkeypatch):
    source = tmp_path / "image.dng"
    source.write_bytes(b"fixture")
    monkeypatch.setitem(sys.modules, "rawpy", None)
    with pytest.raises(ValueError, match="install with pip install"):
        load_frame(source)


def test_dng_rejects_manifest_override(tmp_path):
    source = tmp_path / "image.dng"
    source.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="embedded metadata"):
        load_frame(source, tmp_path / "capture.json")


def test_real_dng_failure_reports_backend_exception(tmp_path):
    pytest.importorskip("rawpy")
    source = tmp_path / "invalid.dng"
    source.write_bytes(b"this is not a DNG")
    with pytest.raises(ValueError, match=r"Cannot decode DNG with rawpy/LibRaw \(LibRaw"):
        load_frame(source)
