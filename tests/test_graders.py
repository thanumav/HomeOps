from homeops.env import HomeOpsEnv
from homeops.graders import grade_episode
from homeops.models import ActionModel


def test_grade_episode_returns_expected_keys():
    env = HomeOpsEnv("saturday_reset")
    env.reset()
    result = grade_episode(env.state())

    expected_keys = {
        "weighted_task_completion",
        "deadline_compliance",
        "readiness_score",
        "satisfaction_score",
        "efficiency_score",
        "final_score",
    }

    assert set(result.keys()) == expected_keys


def test_grade_episode_scores_are_bounded():
    env = HomeOpsEnv("guests_at_6pm")
    env.reset()
    result = grade_episode(env.state())

    for value in result.values():
        assert 0.0 <= value <= 1.0


def test_doing_important_work_improves_grade():
    env = HomeOpsEnv("saturday_reset")
    env.reset()

    before = grade_episode(env.state())["final_score"]

    env.step(
        ActionModel(
            action_type="work_on_task", task_id="pay_electricity_bill", minutes=30
        )
    )
    env.step(ActionModel(action_type="work_on_task", task_id="wash_dishes", minutes=30))

    after = grade_episode(env.state())["final_score"]

    assert after > before


def test_excessive_deferral_hurts_efficiency_score():
    env = HomeOpsEnv("guests_at_6pm")
    env.reset()

    env.step(
        ActionModel(action_type="defer_task", task_id="clean_bathroom", minutes=30)
    )
    env.step(
        ActionModel(
            action_type="defer_task", task_id="clear_living_room_clutter", minutes=30
        )
    )

    result = grade_episode(env.state())

    assert result["efficiency_score"] < 1.0


def test_finished_episode_contains_final_grade():
    env = HomeOpsEnv("saturday_reset")
    env.reset()

    done = False
    info = {}
    while not done:
        pending = [
            t for t in env.state().tasks if t.status in {"pending", "in_progress"}
        ]
        if not pending:
            break
        task = pending[0]
        obs, reward, done, info = env.step(
            ActionModel(action_type="work_on_task", task_id=task.id, minutes=30)
        )

    assert done is True
    assert "final_grade" in info
    assert 0.0 <= info["final_grade"]["final_score"] <= 1.0
