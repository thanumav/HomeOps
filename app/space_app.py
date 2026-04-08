from __future__ import annotations

import json

import gradio as gr

from homeops.env import HomeOpsEnv
from homeops.models import ActionModel


def format_observation(obs) -> str:
    return json.dumps(
        {
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
            "tasks": [task.model_dump() for task in obs.tasks],
            "upcoming_events": [event.model_dump() for event in obs.upcoming_events],
            "last_action_summary": obs.last_action_summary,
            "invalid_action": obs.invalid_action,
        },
        indent=2,
    )


def create_env_state(scenario_id: str):
    env = HomeOpsEnv(scenario_id)
    obs = env.reset()
    return env, format_observation(obs), "Environment reset."


def step_env(env, action_type: str, task_id: str, minutes: int):
    if env is None:
        return None, "", "Please reset the environment first."

    task_id_value = task_id.strip() if task_id else None
    if task_id_value == "":
        task_id_value = None

    try:
        action = ActionModel(
            action_type=action_type,
            task_id=task_id_value,
            minutes=minutes,
        )
        obs, reward, done, info = env.step(action)

        status = {
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }

        return env, format_observation(obs), json.dumps(status, indent=2)
    except Exception as e:
        return env, "", f"Error: {e}"


with gr.Blocks(title="HomeOps") as demo:
    gr.Markdown(
        """
# HomeOps

A household task-management OpenEnv environment for evaluating agents on
prioritization, energy-aware planning, deadline handling, and domestic task execution.
"""
    )

    env_state = gr.State(None)

    with gr.Row():
        scenario = gr.Dropdown(
            choices=HomeOpsEnv.available_scenarios(),
            value="saturday_reset",
            label="Scenario",
        )
        reset_btn = gr.Button("Reset")

    observation_box = gr.Code(label="Observation", language="json")
    status_box = gr.Code(label="Step Result", language="json")

    with gr.Row():
        action_type = gr.Dropdown(
            choices=["work_on_task", "rest", "defer_task"],
            value="work_on_task",
            label="Action Type",
        )
        task_id = gr.Textbox(label="Task ID", placeholder="e.g. pay_electricity_bill")
        minutes = gr.Number(value=30, label="Minutes", precision=0)

    step_btn = gr.Button("Step")

    reset_btn.click(
        fn=create_env_state,
        inputs=[scenario],
        outputs=[env_state, observation_box, status_box],
    )

    step_btn.click(
        fn=step_env,
        inputs=[env_state, action_type, task_id, minutes],
        outputs=[env_state, observation_box, status_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
