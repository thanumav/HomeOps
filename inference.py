from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from openai import OpenAI

from homeops.env import HomeOpsEnv
from homeops.graders import grade_episode
from homeops.models import ActionModel

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
BENCHMARK = "homeops"
SUCCESS_SCORE_THRESHOLD = 0.50


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_system_prompt() -> str:
    return """
You are an agent controlling a structured household-operations environment.

Your job:
- choose the best next action for the current state
- prioritize mandatory, urgent, and deadline-sensitive tasks
- balance energy and stress intelligently
- avoid unnecessary deferrals
- rest only when strategically useful

Return ONLY valid JSON with this exact schema:
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

    return json.dumps(payload, separators=(",", ":"))


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


def fallback_action(obs: Any) -> ActionModel:
    valid_tasks = [
        task
        for task in obs.tasks
        if task.status in {"pending", "in_progress"} and task.dependencies_satisfied
    ]

    if not valid_tasks:
        return ActionModel(action_type="rest", task_id=None, minutes=30)

    valid_tasks = sorted(
        valid_tasks,
        key=lambda t: (
            -int(t.mandatory),
            -(t.priority),
            -(t.urgency),
            t.energy_cost_per_step,
            t.aversion,
        ),
    )

    chosen = valid_tasks[0]
    return ActionModel(action_type="work_on_task", task_id=chosen.id, minutes=30)


def choose_model_action(client: OpenAI, env: HomeOpsEnv) -> ActionModel:
    obs = env._build_observation(invalid_action=False)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(obs)},
        ],
        temperature=0,
        max_tokens=120,
        stream=False,
    )

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty model response")

    return parse_action(text)


def run_task(client: OpenAI | None, scenario_id: str) -> None:
    env = HomeOpsEnv(scenario_id)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=scenario_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        env.reset()
        done = False

        while not done:
            steps_taken += 1

            if client is not None:
                try:
                    action = choose_model_action(client, env)
                except Exception:
                    obs_for_fallback = env._build_observation(invalid_action=False)
                    action = fallback_action(obs_for_fallback)
            else:
                obs_for_fallback = env._build_observation(invalid_action=False)
                action = fallback_action(obs_for_fallback)

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))

            obs, reward, done, info = env.step(action)
            reward_value = reward.value if reward is not None else 0.0

            env_error: Optional[str] = None
            if info.get("invalid_action", False):
                env_error = obs.last_action_summary

            rewards.append(reward_value)

            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward_value,
                done=done,
                error=env_error,
            )

        final_grade = grade_episode(
            env.state(), invalid_action_count=env.invalid_action_count
        )
        score = max(0.0, min(1.0, final_grade["final_score"]))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client: OpenAI | None = None

    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for scenario_id in HomeOpsEnv.available_scenarios():
        run_task(client, scenario_id)


if __name__ == "__main__":
    main()
