from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from homeops.env import HomeOpsEnv
from homeops.graders import grade_episode
from homeops.models import ActionModel

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")


def build_system_prompt() -> str:
    return """
You are an agent controlling a structured household-operations environment.

Your job:
- choose the best next action for the current state
- prioritize mandatory, urgent, and deadline-sensitive tasks
- balance energy and stress intelligently
- avoid unnecessary deferrals
- rest only when strategically useful

You must return ONLY valid JSON with this exact schema:
{
  "action_type": "work_on_task" | "rest" | "defer_task",
  "task_id": "string or null",
  "minutes": 30
}

Rules:
- Use task_id only for work_on_task and defer_task.
- Use task_id as null for rest.
- Always set minutes to 30.
- Do not include markdown.
- Do not include explanations.
""".strip()


def build_user_prompt(obs: Any) -> str:
    tasks = []
    for task in obs.tasks:
        tasks.append(
            {
                "id": task.id,
                "title": task.title,
                "category": task.category,
                "priority": task.priority,
                "urgency": task.urgency,
                "mandatory": task.mandatory,
                "remaining_minutes": task.remaining_minutes,
                "energy_cost_per_step": task.energy_cost_per_step,
                "aversion": task.aversion,
                "deadline_minutes": task.deadline_minutes,
                "status": task.status,
                "dependencies_satisfied": task.dependencies_satisfied,
            }
        )

    events = []
    for event in obs.upcoming_events:
        events.append(
            {
                "id": event.id,
                "title": event.title,
                "event_type": event.event_type,
                "start_minutes": event.start_minutes,
                "end_minutes": event.end_minutes,
                "readiness_metric": event.readiness_metric,
                "readiness_threshold": event.readiness_threshold,
            }
        )

    payload = {
        "current_time_minutes": obs.current_time_minutes,
        "steps_remaining": obs.steps_remaining,
        "energy": obs.energy,
        "stress": obs.stress,
        "satisfaction": obs.satisfaction,
        "cleanliness": obs.cleanliness,
        "clutter": obs.clutter,
        "guest_readiness": obs.guest_readiness,
        "kitchen_readiness": obs.kitchen_readiness,
        "laundry_backlog": obs.laundry_backlog,
        "tasks": tasks,
        "upcoming_events": events,
        "last_action_summary": obs.last_action_summary,
        "invalid_action": obs.invalid_action,
    }

    return json.dumps(payload, indent=2)


def parse_action(response_text: str) -> ActionModel:
    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    data = json.loads(cleaned)
    return ActionModel(**data)


def choose_openai_action(client: OpenAI, env: HomeOpsEnv) -> ActionModel:
    observation = env._build_observation(invalid_action=False)

    response = client.responses.create(
        model=MODEL,
        instructions=build_system_prompt(),
        input=build_user_prompt(observation),
    )

    text = getattr(response, "output_text", "").strip()
    if not text:
        raise ValueError("Model returned empty output.")

    print("raw model output:", repr(text))
    return parse_action(text)


def run_scenario_with_openai(client: OpenAI, scenario_id: str) -> dict[str, Any]:
    env = HomeOpsEnv(scenario_id)
    env.reset()
    print(f"running scenario: {scenario_id}")

    done = False
    info: dict[str, Any] = {}
    total_reward = 0.0
    steps = 0
    actions_taken: list[dict[str, Any]] = []

    while not done:
        try:
            action = choose_openai_action(client, env)
        except Exception as e:
            print(f"model error on scenario={scenario_id}, step={steps + 1}: {e}")
            action = ActionModel(action_type="rest", task_id=None, minutes=30)

        actions_taken.append(action.model_dump())

        _, reward, done, info = env.step(action)
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
        "invalid_action_count": env.invalid_action_count,
        "actions_taken": actions_taken,
        "details": final_grade,
    }


def main():
    print("starting openai baseline")
    print(f"using model: {MODEL}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        print('Set it like this: export OPENAI_API_KEY="your_key_here"')
        raise SystemExit(1)

    client = OpenAI(api_key=api_key)

    scenario_ids = HomeOpsEnv.available_scenarios()
    results = [
        run_scenario_with_openai(client, scenario_id) for scenario_id in scenario_ids
    ]

    print("HomeOps OpenAI baseline results")
    print("=" * 40)
    for result in results:
        print(f"Scenario: {result['scenario_id']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Total reward: {result['total_reward']}")
        print(f"  Final score: {result['final_score']}")
        print(f"  Invalid actions: {result['invalid_action_count']}")
        print()

    average_score = sum(r["final_score"] for r in results) / len(results)
    print(f"Average final score: {average_score:.4f}")


if __name__ == "__main__":
    main()
