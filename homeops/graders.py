from __future__ import annotations

from homeops.models import EnvironmentStateModel
from homeops.utils import task_completion_fraction, weighted_task_score


def compute_weighted_task_completion(state: EnvironmentStateModel) -> float:
    total_weight = 0.0
    achieved_weight = 0.0

    for task in state.tasks:
        weight = weighted_task_score(task)
        total_weight += weight
        achieved_weight += weight * task_completion_fraction(task)

    if total_weight == 0:
        return 0.0
    return achieved_weight / total_weight


def compute_deadline_compliance(state: EnvironmentStateModel) -> float:
    deadline_tasks = [
        task
        for task in state.tasks
        if task.deadline_minutes is not None or task.mandatory
    ]

    if not deadline_tasks:
        return 1.0

    total = 0.0
    for task in deadline_tasks:
        fraction = task_completion_fraction(task)
        if task.status == "missed" and task.mandatory:
            total += 0.0
        elif task.status == "completed":
            total += 1.0
        else:
            total += fraction

    return total / len(deadline_tasks)


def compute_readiness_score(state: EnvironmentStateModel) -> float:
    if state.scenario_id == "guests_at_6pm":
        return state.home.guest_readiness / 100.0

    if state.scenario_id == "overwhelmed_day":
        combined = (
            0.4 * state.home.kitchen_readiness
            + 0.3 * state.home.cleanliness
            + 0.3 * state.home.guest_readiness
        )
        return combined / 100.0

    combined = (
        0.4 * state.home.cleanliness
        + 0.3 * (100.0 - state.home.clutter)
        + 0.3 * state.home.kitchen_readiness
    )
    return combined / 100.0


def compute_satisfaction_score(state: EnvironmentStateModel) -> float:
    return state.user.satisfaction / 100.0


def compute_efficiency_score(
    state: EnvironmentStateModel, invalid_action_count: int = 0
) -> float:
    penalty = 0.0
    penalty += min(0.4, 0.05 * state.user.cumulative_deferrals)
    penalty += min(0.3, 0.08 * state.user.critical_task_deferrals)
    penalty += min(0.3, 0.05 * invalid_action_count)

    return max(0.0, 1.0 - penalty)


def grade_episode(
    state: EnvironmentStateModel, invalid_action_count: int = 0
) -> dict[str, float]:
    weighted_task_completion = compute_weighted_task_completion(state)
    deadline_compliance = compute_deadline_compliance(state)
    readiness_score = compute_readiness_score(state)
    satisfaction_score = compute_satisfaction_score(state)
    efficiency_score = compute_efficiency_score(
        state, invalid_action_count=invalid_action_count
    )

    final_score = (
        0.50 * weighted_task_completion
        + 0.20 * deadline_compliance
        + 0.15 * readiness_score
        + 0.10 * satisfaction_score
        + 0.05 * efficiency_score
    )

    final_score = max(0.0, min(1.0, final_score))

    return {
        "weighted_task_completion": weighted_task_completion,
        "deadline_compliance": deadline_compliance,
        "readiness_score": readiness_score,
        "satisfaction_score": satisfaction_score,
        "efficiency_score": efficiency_score,
        "final_score": final_score,
    }
