from app.space_app import create_env_state


def test_space_app_reset_smoke():
    env, observation, status, suggestion = create_env_state("saturday_reset")
    assert env is not None
    assert "current_time_minutes" in observation
    assert "Environment reset." in status
    assert suggestion == ""
