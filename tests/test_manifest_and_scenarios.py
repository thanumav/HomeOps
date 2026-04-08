from pathlib import Path

from homeops.env import HomeOpsEnv


def test_manifest_exists():
    assert Path("openenv.yaml").exists()


def test_available_scenarios():
    scenarios = HomeOpsEnv.available_scenarios()
    assert "saturday_reset" in scenarios
    assert "guests_at_6pm" in scenarios
    assert "overwhelmed_day" in scenarios
    assert len(scenarios) == 3
