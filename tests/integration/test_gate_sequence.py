"""Integration test: drone passes gates in mock sim with seed 0."""
import pytest

from aigrandprix.config import default_config
from aigrandprix.runner import PipelineRunner


@pytest.mark.integration
def test_completes_all_gates_seed_0():
    """With seed 0 and default config the run completes all gates."""
    cfg = default_config()
    cfg.sim.total_gates = 5
    runner = PipelineRunner(cfg)
    summary = runner.run(seed=0, track_id="gate_sequence")
    assert summary["completion"] is True
    assert summary["gates_passed"] >= 5


@pytest.mark.integration
def test_lap_time_positive_on_completion():
    cfg = default_config()
    cfg.sim.total_gates = 3
    runner = PipelineRunner(cfg)
    summary = runner.run(seed=0, track_id="lap_time")
    assert summary["completion"] is True
    assert summary["lap_time"] > 0.0


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_multiple_seeds_complete():
    """Several different seeds should all complete (robustness check)."""
    cfg = default_config()
    cfg.sim.total_gates = 3
    for seed in (0, 1, 2):
        runner = PipelineRunner(cfg, run_id=f"seed_{seed}")
        summary = runner.run(seed=seed, track_id="multi_seed")
        assert summary["completion"] is True, \
            f"Seed {seed} did not complete: {summary}"
