from __future__ import annotations

from homeops.models import EnvironmentStateModel, RewardModel, TaskItem
from homeops.utils import clamp, critical_pending_tasks


def _task_importance(task: TaskItem) -> float:
    return (
        0.08 * task.priority + 0.06 * task.urgency + (0.05 if task.mandatory else 0.0)
    )


def compute_step_reward(
    previous_state: EnvironmentStateModel,
    current_state: EnvironmentStateModel,
    action_valid: bool,
    action_type: str,
    acted_task: TaskItem | None = None,
    deferred_task: TaskItem | None = None,
    deadline_miss_count: int = 0,
) -> RewardModel:
    components: dict[str, float] = {
        "progress_reward": 0.0,
        "completion_reward": 0.0,
        "state_improvement_reward": 0.0,
        "deadline_alignment_reward": 0.0,
        "anti_procrastination_reward": 0.0,
        "deferral_penalty": 0.0,
        "deadline_miss_penalty": 0.0,
        "waste_penalty": 0.0,
        "exhaustion_penalty": 0.0,
        "invalid_action_penalty": 0.0,
    }

    if not action_valid:
        components["invalid_action_penalty"] = 0.10

    prev_home = previous_state.home
    curr_home = current_state.home
    prev_user = previous_state.user
    curr_user = current_state.user

    # State improvement
    positive_home_gain = 0.0
    positive_home_gain += max(0.0, curr_home.cleanliness - prev_home.cleanliness)
    positive_home_gain += max(0.0, prev_home.clutter - curr_home.clutter)
    positive_home_gain += max(
        0.0, curr_home.guest_readiness - prev_home.guest_readiness
    )
    positive_home_gain += max(
        0.0, curr_home.kitchen_readiness - prev_home.kitchen_readiness
    )
    positive_home_gain += max(
        0.0, prev_home.laundry_backlog - curr_home.laundry_backlog
    )

    components["state_improvement_reward"] = min(0.12, positive_home_gain / 300.0)

    if acted_task is not None and action_valid and action_type == "work_on_task":
        importance = _task_importance(acted_task)

        prev_task = next(t for t in previous_state.tasks if t.id == acted_task.id)
        curr_task = next(t for t in current_state.tasks if t.id == acted_task.id)

        progress_fraction = max(
            0.0,
            (prev_task.remaining_minutes - curr_task.remaining_minutes)
            / max(1, prev_task.estimated_total_minutes),
        )
        components["progress_reward"] = min(0.18, progress_fraction * importance)

        if curr_task.status == "completed":
            components["completion_reward"] = min(
                0.20, 0.04 + importance + curr_task.completion_bonus
            )

        if curr_task.deadline_minutes is not None:
            time_left = (
                curr_task.deadline_minutes - previous_state.user.current_time_minutes
            )
            if time_left <= 180:
                components["deadline_alignment_reward"] = 0.05

        if curr_task.mandatory and curr_task.aversion >= 7:
            prior_deferrals = previous_state.user.critical_task_deferrals
            if prior_deferrals >= 1 or curr_task.deadline_minutes is not None:
                components["anti_procrastination_reward"] = 0.04

    if deferred_task is not None and action_valid and action_type == "defer_task":
        penalty = 0.02
        if (
            deferred_task.mandatory
            or deferred_task.urgency >= 4
            or deferred_task.priority >= 4
        ):
            penalty += 0.04
        components["deferral_penalty"] = penalty

    if deadline_miss_count > 0:
        components["deadline_miss_penalty"] = min(0.30, 0.15 * deadline_miss_count)

    current_critical = critical_pending_tasks(
        current_state.tasks, current_state.user.current_time_minutes
    )
    if action_type == "rest" and current_critical:
        components["waste_penalty"] = 0.04
    elif action_type == "defer_task" and len(current_critical) >= 2:
        components["waste_penalty"] = max(components["waste_penalty"], 0.03)

    if curr_user.energy < 15:
        components["exhaustion_penalty"] = 0.05
    elif curr_user.energy < 5:
        components["exhaustion_penalty"] = 0.10

    raw_value = (
        components["progress_reward"]
        + components["completion_reward"]
        + components["state_improvement_reward"]
        + components["deadline_alignment_reward"]
        + components["anti_procrastination_reward"]
        - components["deferral_penalty"]
        - components["deadline_miss_penalty"]
        - components["waste_penalty"]
        - components["exhaustion_penalty"]
        - components["invalid_action_penalty"]
    )

    value = clamp(raw_value, 0.0, 1.0)
    return RewardModel(value=value, components=components)
