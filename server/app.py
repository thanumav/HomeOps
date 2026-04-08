from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException

from homeops.env import HomeOpsEnv
from homeops.models import ActionModel

app = FastAPI(title="HomeOps", version="0.1.0")

_current_env: HomeOpsEnv | None = None


def get_env() -> HomeOpsEnv:
    global _current_env
    if _current_env is None:
        _current_env = HomeOpsEnv("saturday_reset")
        _current_env.reset()
    return _current_env


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "HomeOps",
        "status": "ok",
        "scenarios": HomeOpsEnv.available_scenarios(),
    }


@app.post("/reset")
def reset(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    global _current_env

    scenario_id = "saturday_reset"
    if payload and "scenario_id" in payload:
        scenario_id = payload["scenario_id"]

    if scenario_id not in HomeOpsEnv.available_scenarios():
        raise HTTPException(
            status_code=400, detail=f"Unknown scenario_id: {scenario_id}"
        )

    _current_env = HomeOpsEnv(scenario_id)
    observation = _current_env.reset()
    return observation.model_dump()


@app.post("/step")
def step(action: ActionModel) -> dict[str, Any]:
    env = get_env()
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    env = get_env()
    return env.state().model_dump()


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
