"""Integration smoke test: full pipeline runs without exceptions."""
import json
from pathlib import Path

import pytest

from aigrandprix.config import default_config
from aigrandprix.runner import PipelineRunner


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_pipeline_smoke_no_exception():
    """Full pipeline runs N steps without raising."""
    cfg = default_config()
    cfg.sim.total_gates = 3     # fewer gates for speed
    cfg.sim.fps = 60
    runner = PipelineRunner(cfg)
    summary = runner.run(seed=0, track_id="smoke_test")
    assert summary is not None


@pytest.mark.integration
def test_pipeline_smoke_log_exists():
    """JSONL log file is created and parseable."""
    cfg = default_config()
    cfg.sim.total_gates = 2
    runner = PipelineRunner(cfg)
    runner.run(seed=1, track_id="log_test")

    assert runner.log_path is not None
    assert runner.log_path.exists()

    lines = runner.log_path.read_text(encoding="utf-8").strip().split("\n")
    # Parse every line
    parsed = [json.loads(line) for line in lines if line.strip()]
    types = [r["type"] for r in parsed]
    assert types[0] == "header"
    assert types[-1] == "footer"
    assert "step" in types


@pytest.mark.integration
def test_pipeline_smoke_footer_has_metrics():
    """Footer contains expected run-level metrics."""
    cfg = default_config()
    cfg.sim.total_gates = 2
    runner = PipelineRunner(cfg)
    runner.run(seed=2, track_id="metrics_test")

    lines = runner.log_path.read_text(encoding="utf-8").strip().split("\n")
    footer = json.loads(lines[-1])
    assert footer["type"] == "footer"
    assert "completion" in footer
    assert "lap_time" in footer
    assert "median_pipeline_ms" in footer
    assert "recovery_events" in footer


@pytest.mark.integration
def test_pipeline_smoke_pipeline_ms_reasonable():
    """Median pipeline_ms should be well under 15ms on modern hardware."""
    cfg = default_config()
    cfg.sim.total_gates = 2
    runner = PipelineRunner(cfg)
    summary = runner.run(seed=3, track_id="timing_test")
    assert summary["median_pipeline_ms"] < 15.0, (
        f"Median pipeline {summary['median_pipeline_ms']:.1f}ms exceeds 15ms budget"
    )
