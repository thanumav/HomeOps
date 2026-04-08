from homeops.env import HomeOpsEnv
from homeops.models import ActionModel


def test_reset_returns_valid_observation():
    env = HomeOpsEnv("saturday_reset")
    obs = env.reset()

    assert obs.current_time_minutes == 8 * 60
    assert len(obs.tasks) == 6
    assert obs.invalid_action is False
    assert obs.last_action_summary == "Episode reset."


def test_work_action_completes_bill_task():
    env = HomeOpsEnv("saturday_reset")
    env.reset()

    obs, reward, done, info = env.step(
        ActionModel(
            action_type="work_on_task", task_id="pay_electricity_bill", minutes=30
        )
    )

    assert obs.current_time_minutes == 8 * 60 + 30
    assert reward.value >= 0.0
    assert done is False
    assert "pay_electricity_bill" in info["completed_tasks"]


def test_invalid_task_action_is_flagged():
    env = HomeOpsEnv("saturday_reset")
    env.reset()

    obs, reward, done, info = env.step(
        ActionModel(action_type="work_on_task", task_id="does_not_exist", minutes=30)
    )

    assert obs.invalid_action is True
    assert info["invalid_action"] is True
    assert reward.value >= 0.0


def test_dependency_blocks_fold_laundry_in_hard_scenario():
    env = HomeOpsEnv("overwhelmed_day")
    env.reset()

    obs, reward, done, info = env.step(
        ActionModel(action_type="work_on_task", task_id="fold_laundry", minutes=30)
    )

    assert obs.invalid_action is True
    assert info["invalid_action"] is True


def test_episode_can_finish():
    env = HomeOpsEnv("saturday_reset")
    env.reset()

    done = False
    while not done:
        pending = [
            t for t in env.state().tasks if t.status in {"pending", "in_progress"}
        ]
        if pending:
            task_id = pending[0].id
            obs, reward, done, info = env.step(
                ActionModel(action_type="work_on_task", task_id=task_id, minutes=30)
            )
        else:
            break

    assert done is True
    assert "final_grade" in info
