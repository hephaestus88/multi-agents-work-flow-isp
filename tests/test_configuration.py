import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import pytest

from isp_workflow.agents import QualityPolicy
from isp_workflow.models import RawFrame, RenderConfig
from isp_workflow.workflow import Coordinator, load_configuration


@pytest.mark.parametrize("field", ["exposure_ev", "contrast", "denoise", "sharpen"])
def test_render_boolean_is_not_numeric(field):
    with pytest.raises(ValueError):
        RenderConfig(**{field: True})


def test_quality_boolean_is_not_numeric():
    with pytest.raises(ValueError):
        QualityPolicy(max_clipped_fraction=True)


def test_duplicate_configuration_keys_rejected(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"quality":{"max_clipped_fraction":0,"max_clipped_fraction":1}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_configuration(config)


def test_configuration_bom_and_unknown_keys(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"candidates": [{"name": "one"}]}), encoding="utf-8-sig")
    candidates, _ = load_configuration(config)
    assert candidates[0].name == "one"
    config.write_text('{"typo":true}')
    with pytest.raises(ValueError):
        load_configuration(config)


def test_output_directory_reserved_before_agents_run(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class BlockingIntake:
        def run(self, input_path, manifest_path):
            entered.set()
            assert release.wait(timeout=10)
            return RawFrame(np.full((2, 2, 3), 0.2, np.float32), "linear_srgb")

    output = tmp_path / "run"
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(Coordinator(intake=BlockingIntake()).run, Path("unused"), output)
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(ValueError, match="existing runs|reserved"):
                Coordinator().run("unused", output)
        finally:
            release.set()
        assert first.result(timeout=10).output_path.exists()
    assert not (output / ".run.lock").exists()
