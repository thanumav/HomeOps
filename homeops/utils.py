from __future__ import annotations

from homeops.models import EventItem, TaskItem


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def minutes_to_time_str(total_minutes: int) -> str:
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def get_task_by_id(tasks: list[TaskItem], task_id: str) -> TaskItem | None:
    for task in tasks:
        if task.id == task_id:
            return task
    return None


def get_event_by_id(events: list[EventItem], event_id: str) -> EventItem | None:
    for event in events:
        if event.id == event_id:
            return event
    return None


def is_task_finished(task: TaskItem) -> bool:
    return task.status in {"completed", "missed", "blocked"}


def dependencies_satisfied(task: TaskItem, tasks: list[TaskItem]) -> bool:
    if not task.dependencies:
        return True

    completed_task_ids = {t.id for t in tasks if t.status == "completed"}
    return all(dep_id in completed_task_ids for dep_id in task.dependencies)


def pending_tasks(tasks: list[TaskItem]) -> list[TaskItem]:
    return [task for task in tasks if task.status in {"pending", "in_progress"}]


def completed_tasks(tasks: list[TaskItem]) -> list[TaskItem]:
    return [task for task in tasks if task.status == "completed"]


def missed_tasks(tasks: list[TaskItem]) -> list[TaskItem]:
    return [task for task in tasks if task.status == "missed"]


def critical_pending_tasks(
    tasks: list[TaskItem], current_time_minutes: int
) -> list[TaskItem]:
    critical = []
    for task in tasks:
        if task.status not in {"pending", "in_progress"}:
            continue

        if task.mandatory:
            critical.append(task)
            continue

        if task.deadline_minutes is not None:
            minutes_left = task.deadline_minutes - current_time_minutes
            if minutes_left <= 120:
                critical.append(task)
                continue

        if task.priority >= 5 or task.urgency >= 5:
            critical.append(task)

    return critical


def task_completion_fraction(task: TaskItem) -> float:
    if task.estimated_total_minutes <= 0:
        return 0.0
    completed = task.estimated_total_minutes - task.remaining_minutes
    return max(0.0, min(1.0, completed / task.estimated_total_minutes))


def weighted_task_score(task: TaskItem) -> float:
    return 0.5 * task.priority + 0.3 * task.urgency + 0.2 * int(task.mandatory)
