"""Workflow checks exercise real artifacts, failure gates, and extension points."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from isp_workflow.agents import PresetTuningPolicy, QualityPolicy
from isp_workflow.cli import main
from isp_workflow.models import RenderConfig
from isp_workflow.workflow import Coordinator, QualityGateError


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def raw_capture(tmp_path):
    # Each published RAW10 group decodes to four samples of 0x155 = 341.
    source = tmp_path / "capture.raw"
    source.write_bytes(bytes([0x55] * 5) * 2 * 6)
    manifest = tmp_path / "capture.json"
    manifest.write_text(json.dumps({
        "layout": "mipi_raw10", "width": 8, "height": 6, "row_stride_bytes": 10,
        "cfa_pattern_at_frame_origin": "RGGB", "black_level": 0,
        "white_level": 1023, "white_balance": [1, 1, 1],
        "camera_to_linear_srgb": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "crop": [1, 1, 5, 3],
    }), encoding="utf-8")
    return source, manifest


def test_raw_to_verified_24bit_bmp_and_artifacts(tmp_path, raw_capture):
    source, manifest = raw_capture
    output = tmp_path / "run"
    result = Coordinator().run(source, output, manifest_path=manifest, save_intermediates=True)
    assert result.output_path == output / "output.bmp"
    header = result.output_path.read_bytes()
    assert header[:2] == b"BM"
    assert int.from_bytes(header[28:30], "little") == 24
    with Image.open(result.output_path) as image:
        assert image.size == (5, 3)  # Width exercises BMP row padding.
        assert image.mode == "RGB"
        # Linear 1/3 encodes to sRGB code 156.
        np.testing.assert_array_equal(np.asarray(image), np.full((3, 5, 3), 156, dtype=np.uint8))
    prepared = np.load(output / "intermediate.npy", allow_pickle=False)
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, 1 / 3, atol=1e-6)
    report = read_json(output / "quality_report.json")
    assert report["status"] == "passed_automatic_checks"
    assert report["export"]["round_trip_verified"] is True
    assert report["export"]["bits_per_pixel"] == 24
    selected = next(item for item in report["candidates"] if item["candidate"] == result.selected_candidate)
    with Image.open(output / selected["preview"]) as image:
        np.testing.assert_array_equal(np.asarray(image), np.full((3, 5, 3), 156, dtype=np.uint8))
    events = read_json(output / "agent_log.json")
    assert {item["agent"] for item in events} == {"coordinator", "intake", "isp", "tuning", "quality", "export"}
    assert events[-1]["action"] == "completed"


def test_artifacts_record_input_and_runtime_provenance(tmp_path, raw_capture):
    source, manifest = raw_capture
    output = tmp_path / "run"
    Coordinator().run(source, output, manifest_path=manifest)
    metadata = read_json(output / "metadata.json")
    assert metadata["metadata"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["metadata"]["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert metadata["metadata"]["resolved_manifest"]["crop"] == [1, 1, 5, 3]
    assert metadata["working_image"] == {"shape": [3, 5, 3], "dtype": "float32", "space": "linear_srgb"}
    assert set(metadata["runtime"]) >= {"python", "numpy", "pillow", "workflow"}
    configuration = read_json(output / "isp_config.json")
    assert configuration["selected"]["name"] == "baseline"
    assert configuration["quality"]["max_clipped_fraction"] == 0.1
    assert not (output / "intermediate.npy").exists()


def test_all_candidates_rejected_retains_diagnostics_without_bmp(tmp_path, raw_capture):
    source, manifest = raw_capture
    # Every site becomes sensor white; a baseline-only policy must fail the gate.
    source.write_bytes(bytes([0xFF] * 5) * 2 * 6)
    output = tmp_path / "rejected"
    workflow = Coordinator(tuning_policy=PresetTuningPolicy([RenderConfig()]),
                           quality_policy=QualityPolicy(max_clipped_fraction=0))
    with pytest.raises(QualityGateError, match="All candidates failed"):
        workflow.run(source, output, manifest_path=manifest)
    assert not (output / "output.bmp").exists()
    assert (output / "previews" / "candidate_00.png").is_file()
    report = read_json(output / "quality_report.json")
    assert report["status"] == "failed"
    assert report["selected"] is None
    assert report["candidates"][0]["clipped_pixel_fraction"] == 1
    assert read_json(output / "failure.json")["error"] == "QualityGateError"
    assert read_json(output / "agent_log.json")[-1]["action"] == "failed"


def test_custom_tuning_policy_gets_read_only_prepared_frame(tmp_path, raw_capture):
    class CustomPolicy:
        def __init__(self):
            self.called = 0

        def propose(self, frame, linear):
            self.called += 1
            assert frame.representation == "bayer"
            assert linear.flags.writeable is False
            np.testing.assert_allclose(linear, 1 / 3, atol=1e-6)
            return [RenderConfig(name="custom", exposure_ev=-1)]

    policy = CustomPolicy()
    source, manifest = raw_capture
    output = tmp_path / "custom"
    result = Coordinator(tuning_policy=policy).run(source, output, manifest_path=manifest)
    assert policy.called == 1
    assert result.selected_candidate == "custom"
    with Image.open(result.output_path) as image:
        assert int(np.asarray(image)[0, 0, 0]) == 113


def test_parallel_candidate_evaluation_is_deterministic(tmp_path, raw_capture):
    source, manifest = raw_capture
    results = []
    for workers in (1, 2):
        output = tmp_path / f"workers_{workers}"
        result = Coordinator().run(source, output, manifest_path=manifest, workers=workers)
        results.append((output, result))
    first, second = results
    assert first[1].selected_candidate == second[1].selected_candidate
    assert first[1].output_path.read_bytes() == second[1].output_path.read_bytes()
    assert read_json(first[0] / "quality_report.json") == read_json(second[0] / "quality_report.json")
    for preview in (first[0] / "previews").glob("*.png"):
        assert preview.read_bytes() == (second[0] / "previews" / preview.name).read_bytes()


def test_existing_run_is_preserved(tmp_path, raw_capture):
    source, manifest = raw_capture
    output = tmp_path / "run"
    result = Coordinator().run(source, output, manifest_path=manifest)
    original = result.output_path.read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        Coordinator().run(source, output, manifest_path=manifest)
    assert result.output_path.read_bytes() == original


def test_cli_missing_manifest_has_concise_error(tmp_path, raw_capture, capsys):
    source, _ = raw_capture
    status = main([str(source), "--output-dir", str(tmp_path / "missing_manifest")])
    captured = capsys.readouterr()
    assert status == 1
    assert "require --manifest" in captured.err
    assert "Traceback" not in captured.err


def test_cli_quality_failure_has_distinct_exit_code(tmp_path, raw_capture, capsys):
    source, manifest = raw_capture
    source.write_bytes(bytes([0xFF] * 5) * 2 * 6)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"candidates": [{"name": "baseline"}],
                                  "quality": {"max_clipped_fraction": 0}}))
    status = main([str(source), "--manifest", str(manifest), "--output-dir",
                   str(tmp_path / "gate"), "--config", str(config)])
    captured = capsys.readouterr()
    assert status == 2
    assert "clipping gate" in captured.err
    assert "Traceback" not in captured.err


def test_module_entry_point_converts_capture(tmp_path, raw_capture):
    source, manifest = raw_capture
    output = tmp_path / "cli"
    result = subprocess.run([sys.executable, "-m", "isp_workflow", str(source),
                             "--manifest", str(manifest), "--output-dir", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "BMP:" in result.stdout
    assert "Traceback" not in result.stderr
    assert (output / "output.bmp").is_file()


def test_real_generated_dng_converts_via_rawpy(tmp_path):
    pytest.importorskip("rawpy")
    pytest.importorskip("tifffile")
    script = Path(__file__).resolve().parents[1] / "examples" / "generate_demo.py"
    spec = importlib.util.spec_from_file_location("isp_demo_generator", script)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    source = tmp_path / "synthetic.dng"
    demo.write_dng(source, demo.mosaic(demo.scene(32, 24), 256, 4095), 256, 4095)
    output = tmp_path / "dng"
    result = Coordinator().run(source, output)
    with Image.open(result.output_path) as image:
        assert image.size == (32, 24)
        assert image.mode == "RGB"
        assert np.asarray(image).std() > 10
    metadata = read_json(output / "metadata.json")
    assert metadata["representation"] == "linear_srgb"
    assert metadata["metadata"]["backend"] == "rawpy/LibRaw"
    assert metadata["metadata"]["backend_settings"]["gamma"] == [1.0, 1.0]
    assert "backend_color_conversion" in metadata["stages"]
    assert read_json(output / "quality_report.json")["export"]["round_trip_verified"] is True
