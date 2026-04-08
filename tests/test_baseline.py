from baseline.run_baseline import run_scenario


def test_baseline_runs_on_easy_scenario():
    result = run_scenario("saturday_reset")
    assert result["scenario_id"] == "saturday_reset"
    assert result["steps"] > 0
    assert 0.0 <= result["final_score"] <= 1.0
