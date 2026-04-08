from __future__ import annotations

from copy import deepcopy

from homeops.graders import grade_episode
from homeops.models import (
    ActionModel,
    EnvironmentStateModel,
    EventObservation,
    ObservationModel,
    ScenarioConfig,
    TaskObservation,
)
from homeops.reward import compute_step_reward
from homeops.scenarios import SCENARIOS
from homeops.utils import (
    clamp,
    completed_tasks,
    dependencies_satisfied,
    get_task_by_id,
    is_task_finished,
    minutes_to_time_str,
    missed_tasks,
    pending_tasks,
)


class HomeOpsEnv:
    def __init__(self, scenario_id: str = "saturday_reset"):
        if scenario_id not in SCENARIOS:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")

        self.scenario_id = scenario_id
        self.scenario_factory = SCENARIOS[scenario_id]
        self.config: ScenarioConfig | None = None
        self._state: EnvironmentStateModel | None = None
        self.invalid_action_count: int = 0
        self.last_action_summary: str = "Environment initialized."

    @staticmethod
    def available_scenarios() -> list[str]:
        return list(SCENARIOS.keys())

    def reset(self) -> ObservationModel:
        self.config = self.scenario_factory()
        self.invalid_action_count = 0
        self.last_action_summary = "Episode reset."

        self._state = EnvironmentStateModel(
            scenario_id=self.config.id,
            day_start_minutes=self.config.day_start_minutes,
            day_end_minutes=self.config.day_end_minutes,
            step_minutes=self.config.step_minutes,
            user=deepcopy(self.config.initial_user),
            home=deepcopy(self.config.initial_home),
            tasks=deepcopy(self.config.tasks),
            events=deepcopy(self.config.events),
            step_count=0,
            done=False,
        )

        return self._build_observation(invalid_action=False)

    def state(self) -> EnvironmentStateModel:
        if self._state is None:
            raise RuntimeError("Environment has not been reset.")
        return self._state

    def step(self, action: ActionModel):
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        if self._state.done:
            raise RuntimeError("Episode is already done.")

        previous_state = deepcopy(self._state)
        action_valid = True
        acted_task = None
        deferred_task = None
        deadline_miss_count = 0

        if action.minutes != self._state.step_minutes:
            action_valid = False
            self.last_action_summary = f"Invalid action: minutes must equal step size {self._state.step_minutes}."
            self.invalid_action_count += 1
        else:
            if action.action_type == "work_on_task":
                action_valid, acted_task = self._apply_work_action(action.task_id)
            elif action.action_type == "rest":
                action_valid = self._apply_rest_action()
            elif action.action_type == "defer_task":
                action_valid, deferred_task = self._apply_defer_action(action.task_id)
            else:
                action_valid = False
                self.last_action_summary = f"Invalid action type: {action.action_type}"
                self.invalid_action_count += 1

        self._advance_time()

        deadline_miss_count = self._update_deadlines_and_events()

        reward = compute_step_reward(
            previous_state=previous_state,
            current_state=self._state,
            action_valid=action_valid,
            action_type=action.action_type,
            acted_task=acted_task,
            deferred_task=deferred_task,
            deadline_miss_count=deadline_miss_count,
        )

        self._state.step_count += 1
        self._state.done = self._check_done()

        info = {
            "scenario_id": self._state.scenario_id,
            "reward_components": reward.components,
            "completed_tasks": [task.id for task in completed_tasks(self._state.tasks)],
            "missed_tasks": [task.id for task in missed_tasks(self._state.tasks)],
            "invalid_action": not action_valid,
            "time_str": minutes_to_time_str(self._state.user.current_time_minutes),
        }

        if self._state.done:
            info["final_grade"] = grade_episode(
                self._state,
                invalid_action_count=self.invalid_action_count,
            )

        observation = self._build_observation(invalid_action=not action_valid)
        return observation, reward, self._state.done, info

    def _apply_work_action(self, task_id: str | None):
        if task_id is None:
            self.last_action_summary = (
                "Invalid action: task_id is required for work_on_task."
            )
            self.invalid_action_count += 1
            return False, None

        task = get_task_by_id(self._state.tasks, task_id)
        if task is None:
            self.last_action_summary = f"Invalid action: task '{task_id}' not found."
            self.invalid_action_count += 1
            return False, None

        if is_task_finished(task):
            self.last_action_summary = (
                f"Invalid action: task '{task_id}' is already finished."
            )
            self.invalid_action_count += 1
            return False, None

        if not dependencies_satisfied(task, self._state.tasks):
            self.last_action_summary = (
                f"Invalid action: dependencies for '{task_id}' are not satisfied."
            )
            self.invalid_action_count += 1
            return False, None

        if not task.can_split and task.remaining_minutes < task.estimated_total_minutes:
            self.last_action_summary = (
                f"Invalid action: task '{task_id}' cannot be split."
            )
            self.invalid_action_count += 1
            return False, None

        task.status = "in_progress"

        work_minutes = min(self._state.step_minutes, task.remaining_minutes)
        task.remaining_minutes = max(0, task.remaining_minutes - work_minutes)

        self._state.user.minutes_spent_working += self._state.step_minutes
        self._state.user.energy = clamp(
            self._state.user.energy - task.energy_cost_per_step
        )
        self._state.user.stress = clamp(
            self._state.user.stress + max(0.0, task.aversion * 0.8 - 2.0),
        )

        if task.remaining_minutes == 0:
            task.status = "completed"
            self._state.user.completed_task_count += 1
            self._state.user.satisfaction = clamp(
                self._state.user.satisfaction + task.satisfaction_impact + 3.0
            )
            self._apply_task_impacts(task)
            self.last_action_summary = f"Completed task: {task.title}"
        else:
            self.last_action_summary = f"Worked on task: {task.title}"

        return True, task

    def _apply_rest_action(self) -> bool:
        self._state.user.minutes_spent_resting += self._state.step_minutes
        self._state.user.energy = clamp(self._state.user.energy + 12.0)
        self._state.user.stress = clamp(self._state.user.stress - 6.0)
        self._state.user.satisfaction = clamp(self._state.user.satisfaction + 1.0)
        self.last_action_summary = "Rested for one step."
        return True

    def _apply_defer_action(self, task_id: str | None):
        if task_id is None:
            self.last_action_summary = (
                "Invalid action: task_id is required for defer_task."
            )
            self.invalid_action_count += 1
            return False, None

        task = get_task_by_id(self._state.tasks, task_id)
        if task is None:
            self.last_action_summary = f"Invalid action: task '{task_id}' not found."
            self.invalid_action_count += 1
            return False, None

        if is_task_finished(task):
            self.last_action_summary = (
                f"Invalid action: task '{task_id}' is already finished."
            )
            self.invalid_action_count += 1
            return False, None

        self._state.user.cumulative_deferrals += 1
        if task.mandatory or task.priority >= 4 or task.urgency >= 4:
            self._state.user.critical_task_deferrals += 1

        self._state.user.stress = clamp(self._state.user.stress + 3.0)
        self._state.user.satisfaction = clamp(self._state.user.satisfaction - 1.0)
        self.last_action_summary = f"Deferred task: {task.title}"

        return True, task

    def _apply_task_impacts(self, task):
        self._state.home.cleanliness = clamp(
            self._state.home.cleanliness + task.cleanliness_impact
        )
        self._state.home.clutter = clamp(self._state.home.clutter + task.clutter_impact)
        self._state.home.guest_readiness = clamp(
            self._state.home.guest_readiness + task.guest_readiness_impact
        )
        self._state.home.kitchen_readiness = clamp(
            self._state.home.kitchen_readiness + task.kitchen_readiness_impact
        )
        self._state.home.laundry_backlog = clamp(
            self._state.home.laundry_backlog + task.laundry_backlog_impact
        )

    def _advance_time(self):
        self._state.user.current_time_minutes += self._state.step_minutes

    def _update_deadlines_and_events(self) -> int:
        deadline_miss_count = 0
        now = self._state.user.current_time_minutes

        for task in self._state.tasks:
            if task.status in {"completed", "missed", "blocked"}:
                continue

            if task.deadline_minutes is not None and now > task.deadline_minutes:
                task.status = "missed"
                deadline_miss_count += 1
                self._state.user.stress = clamp(self._state.user.stress + 12.0)
                self._state.user.satisfaction = clamp(
                    self._state.user.satisfaction - 8.0
                )

        for event in self._state.events:
            if event.readiness_metric == "none" or event.readiness_threshold is None:
                continue

            if now >= event.start_minutes:
                current_metric_value = getattr(self._state.home, event.readiness_metric)
                if current_metric_value < event.readiness_threshold:
                    self._state.user.stress = clamp(self._state.user.stress + 8.0)
                    self._state.user.satisfaction = clamp(
                        self._state.user.satisfaction - 6.0
                    )

        return deadline_miss_count

    def _check_done(self) -> bool:
        if self._state.user.current_time_minutes >= self._state.day_end_minutes:
            return True

        if not pending_tasks(self._state.tasks):
            return True

        return False

    def _build_observation(self, invalid_action: bool) -> ObservationModel:
        current_time = self._state.user.current_time_minutes
        steps_remaining = max(
            0,
            (self._state.day_end_minutes - current_time) // self._state.step_minutes,
        )

        task_observations = []
        for task in self._state.tasks:
            task_observations.append(
                TaskObservation(
                    id=task.id,
                    title=task.title,
                    category=task.category,
                    priority=task.priority,
                    urgency=task.urgency,
                    mandatory=task.mandatory,
                    remaining_minutes=task.remaining_minutes,
                    energy_cost_per_step=task.energy_cost_per_step,
                    aversion=task.aversion,
                    deadline_minutes=task.deadline_minutes,
                    status=task.status,
                    dependencies_satisfied=dependencies_satisfied(
                        task, self._state.tasks
                    ),
                )
            )

        upcoming_events = []
        for event in self._state.events:
            if event.start_minutes >= current_time:
                upcoming_events.append(
                    EventObservation(
                        id=event.id,
                        title=event.title,
                        event_type=event.event_type,
                        start_minutes=event.start_minutes,
                        end_minutes=event.end_minutes,
                        readiness_metric=event.readiness_metric,
                        readiness_threshold=event.readiness_threshold,
                    )
                )

        return ObservationModel(
            current_time_minutes=current_time,
            steps_remaining=steps_remaining,
            energy=self._state.user.energy,
            stress=self._state.user.stress,
            satisfaction=self._state.user.satisfaction,
            cleanliness=self._state.home.cleanliness,
            clutter=self._state.home.clutter,
            guest_readiness=self._state.home.guest_readiness,
            kitchen_readiness=self._state.home.kitchen_readiness,
            laundry_backlog=self._state.home.laundry_backlog,
            tasks=task_observations,
            upcoming_events=upcoming_events,
            last_action_summary=self.last_action_summary,
            invalid_action=invalid_action,
        )
