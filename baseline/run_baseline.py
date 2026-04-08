from __future__ import annotations

from homeops.env import HomeOpsEnv
from homeops.graders import grade_episode
from homeops.models import ActionModel
from homeops.utils import dependencies_satisfied


def choose_heuristic_action(env: HomeOpsEnv) -> ActionModel:
    state = env.state()
    tasks = [t for t in state.tasks if t.status in {"pending", "in_progress"}]
    valid_tasks = [t for t in tasks if dependencies_satisfied(t, state.tasks)]

    if not valid_tasks:
        return ActionModel(action_type="rest", minutes=30)

    current_time = state.user.current_time_minutes

    def minutes_until_next_guest_event() -> int | None:
        guest_events = [
            e
            for e in state.events
            if e.event_type == "guests_arrival" and e.start_minutes >= current_time
        ]
        if not guest_events:
            return None
        return min(e.start_minutes - current_time for e in guest_events)

    guest_minutes_left = minutes_until_next_guest_event()

    def guest_relevance_score(task) -> float:
        score = 0.0

        # Direct environment impacts
        score += task.guest_readiness_impact * 2.0
        score += task.cleanliness_impact * 0.8
        score += (-task.clutter_impact) * 0.8 if task.clutter_impact < 0 else 0.0
        score += task.kitchen_readiness_impact * 0.5

        # Human-like guest-facing boosts
        title = task.title.lower()
        room = (task.room or "").lower()

        if "bathroom" in title or room == "bathroom":
            score += 18.0
        if "living room" in title or room == "living_room":
            score += 12.0
        if "trash" in title:
            score += 10.0
        if "guest" in title or task.category == "guest_prep":
            score += 12.0

        return score

    def sort_key(task):
        deadline_pressure = 999999
        if task.deadline_minutes is not None:
            deadline_pressure = task.deadline_minutes - current_time

        guest_priority = 0.0
        if (
            guest_minutes_left is not None and guest_minutes_left <= 600
        ):  # within 10 hours
            guest_priority = guest_relevance_score(task)

            # Make guest pressure sharper as the visit gets closer
            if guest_minutes_left <= 240:  # within 4 hours
                guest_priority *= 1.5
            if guest_minutes_left <= 120:  # within 2 hours
                guest_priority *= 1.8

        return (
            -int(task.mandatory),
            deadline_pressure,
            -task.priority,
            -task.urgency,
            -guest_priority,
            task.energy_cost_per_step,
            task.aversion,
        )

    if state.user.energy < 15:
        urgent_exists = any(
            t.mandatory
            or (
                t.deadline_minutes is not None
                and t.deadline_minutes - current_time <= 60
            )
            for t in valid_tasks
        )
        if not urgent_exists:
            return ActionModel(action_type="rest", minutes=30)

    chosen = sorted(valid_tasks, key=sort_key)[0]
    return ActionModel(action_type="work_on_task", task_id=chosen.id, minutes=30)


def run_scenario(scenario_id: str) -> dict:
    env = HomeOpsEnv(scenario_id)
    env.reset()

    done = False
    info = {}
    total_reward = 0.0
    steps = 0

    while not done:
        action = choose_heuristic_action(env)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

    final_grade = info.get(
        "final_grade",
        grade_episode(env.state(), invalid_action_count=env.invalid_action_count),
    )

    return {
        "scenario_id": scenario_id,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "final_score": round(final_grade["final_score"], 4),
        "details": final_grade,
    }


def main():
    scenario_ids = HomeOpsEnv.available_scenarios()
    results = [run_scenario(scenario_id) for scenario_id in scenario_ids]

    print("HomeOps heuristic baseline results")
    print("=" * 40)
    for result in results:
        print(f"Scenario: {result['scenario_id']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Total reward: {result['total_reward']}")
        print(f"  Final score: {result['final_score']}")
        print()

    average_score = sum(r["final_score"] for r in results) / len(results)
    print(f"Average final score: {average_score:.4f}")


if __name__ == "__main__":
    main()
