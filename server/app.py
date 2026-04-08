from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>HomeOps</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --border: #e5e7eb;
      --accent: #2563eb;
      --chip: #eef4ff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }

    .wrap {
      max-width: 900px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }

    .hero, .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 8px 30px rgba(16, 24, 40, 0.05);
    }

    .hero {
      padding: 32px;
    }

    .badge {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--chip);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }

    h1 {
      margin: 0 0 10px;
      font-size: 40px;
      line-height: 1.05;
    }

    .sub {
      margin: 0;
      color: var(--muted);
      font-size: 17px;
      max-width: 720px;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }

    .chip {
      border: 1px solid var(--border);
      background: #fafafa;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 14px;
      color: var(--text);
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
      margin-top: 20px;
    }

    .card {
      grid-column: span 12;
      padding: 22px;
    }

    .half {
      grid-column: span 6;
    }

    h2 {
      margin: 0 0 10px;
      font-size: 21px;
    }

    p, li {
      color: var(--muted);
      margin-top: 0;
    }

    ul {
      margin: 10px 0 0 18px;
      padding: 0;
    }

    code {
      background: #f3f4f6;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 2px 6px;
      color: #111827;
      font-size: 0.95em;
    }

    a {
      color: var(--accent);
      text-decoration: none;
    }

    a:hover {
      text-decoration: underline;
    }

    .footer {
      text-align: center;
      color: var(--muted);
      font-size: 14px;
      margin-top: 20px;
    }

    @media (max-width: 800px) {
      h1 { font-size: 32px; }
      .half { grid-column: span 12; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="badge">OpenEnv Environment</div>
      <h1>HomeOps</h1>
      <p class="sub">
        A household task-management environment for evaluating agents on prioritization,
        deadline handling, energy-aware planning, and domestic task execution.
      </p>

      <div class="chips">
        <div class="chip">Real-world simulation</div>
        <div class="chip">Typed models</div>
        <div class="chip">3 benchmark scenarios</div>
        <div class="chip">Deterministic grading</div>
        <div class="chip">Dense rewards</div>
      </div>
    </section>

    <section class="grid">
      <div class="card half">
        <h2>What it does</h2>
        <p>
          HomeOps simulates a single day of household decision-making. An agent must choose
          what to do next while balancing urgency, energy, aversion, deadlines, and home readiness.
        </p>
      </div>

      <div class="card half">
        <h2>Scenarios</h2>
        <ul>
          <li><strong>Saturday Reset</strong> — easy</li>
          <li><strong>Guests at 6 PM</strong> — medium</li>
          <li><strong>Overwhelmed Day</strong> — hard</li>
        </ul>
      </div>

      <div class="card half">
        <h2>API</h2>
        <ul>
          <li><code>GET /</code> — overview page</li>
          <li><code>POST /reset</code> — reset environment</li>
          <li><code>POST /step</code> — apply an action</li>
          <li><code>GET /state</code> — current full state</li>
        </ul>
      </div>

      <div class="card half">
        <h2>Docs</h2>
        <p>
          Interactive API docs are available at
          <a href="/docs">/docs</a>.
        </p>
        <p>
          OpenAPI schema is available at
          <a href="/openapi.json">/openapi.json</a>.
        </p>
      </div>

      <div class="card">
        <h2>Action format</h2>
        <p>
          Actions are structured JSON objects with
          <code>action_type</code>, optional <code>task_id</code>, and fixed
          <code>minutes=30</code>.
        </p>
      </div>
    </section>

    <div class="footer">
      HomeOps • household operations benchmark
    </div>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return LANDING_PAGE


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
