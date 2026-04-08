from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


TASK_CATEGORIES = Literal[
    "cleaning",
    "laundry",
    "kitchen",
    "errand",
    "admin",
    "guest_prep",
    "maintenance",
]

TASK_STATUS = Literal[
    "pending",
    "in_progress",
    "completed",
    "missed",
    "blocked",
]

EVENT_TYPES = Literal[
    "guests_arrival",
    "delivery_window",
    "bill_cutoff",
    "trash_pickup",
    "appointment",
]

READINESS_METRICS = Literal[
    "guest_readiness",
    "kitchen_readiness",
    "cleanliness",
    "none",
]

ACTION_TYPES = Literal[
    "work_on_task",
    "rest",
    "defer_task",
]


class TaskItem(BaseModel):
    id: str
    title: str
    category: TASK_CATEGORIES
    room: Optional[str] = None

    priority: int = Field(..., ge=1, le=5)
    urgency: int = Field(..., ge=1, le=5)
    mandatory: bool = False

    estimated_total_minutes: int = Field(..., gt=0)
    remaining_minutes: int = Field(..., ge=0)
    min_work_chunk_minutes: int = Field(default=30, gt=0)
    can_split: bool = True

    energy_cost_per_step: int = Field(..., ge=0, le=30)
    aversion: int = Field(..., ge=0, le=10)

    deadline_minutes: Optional[int] = None
    status: TASK_STATUS = "pending"

    dependencies: list[str] = Field(default_factory=list)
    completion_bonus: float = 0.0

    cleanliness_impact: float = 0.0
    clutter_impact: float = 0.0
    guest_readiness_impact: float = 0.0
    kitchen_readiness_impact: float = 0.0
    laundry_backlog_impact: float = 0.0
    satisfaction_impact: float = 0.0

    @field_validator("remaining_minutes")
    @classmethod
    def remaining_not_more_than_total(cls, value: int, info) -> int:
        total = info.data.get("estimated_total_minutes")
        if total is not None and value > total:
            raise ValueError("remaining_minutes cannot exceed estimated_total_minutes")
        return value


class EventItem(BaseModel):
    id: str
    title: str
    event_type: EVENT_TYPES
    start_minutes: int = Field(..., ge=0)
    end_minutes: Optional[int] = Field(default=None, ge=0)
    mandatory: bool = True

    related_task_ids: list[str] = Field(default_factory=list)
    readiness_metric: READINESS_METRICS = "none"
    readiness_threshold: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    @field_validator("end_minutes")
    @classmethod
    def end_not_before_start(cls, value: Optional[int], info) -> Optional[int]:
        start = info.data.get("start_minutes")
        if value is not None and start is not None and value < start:
            raise ValueError("end_minutes cannot be earlier than start_minutes")
        return value


class UserState(BaseModel):
    current_time_minutes: int = Field(..., ge=0)
    energy: float = Field(..., ge=0.0, le=100.0)
    stress: float = Field(..., ge=0.0, le=100.0)
    satisfaction: float = Field(..., ge=0.0, le=100.0)

    cumulative_deferrals: int = Field(default=0, ge=0)
    critical_task_deferrals: int = Field(default=0, ge=0)
    completed_task_count: int = Field(default=0, ge=0)

    minutes_spent_resting: int = Field(default=0, ge=0)
    minutes_spent_working: int = Field(default=0, ge=0)


class HomeState(BaseModel):
    cleanliness: float = Field(..., ge=0.0, le=100.0)
    clutter: float = Field(..., ge=0.0, le=100.0)
    guest_readiness: float = Field(..., ge=0.0, le=100.0)
    kitchen_readiness: float = Field(..., ge=0.0, le=100.0)
    laundry_backlog: float = Field(..., ge=0.0, le=100.0)


class TaskObservation(BaseModel):
    id: str
    title: str
    category: str
    priority: int
    urgency: int
    mandatory: bool
    remaining_minutes: int
    energy_cost_per_step: int
    aversion: int
    deadline_minutes: Optional[int] = None
    status: str
    dependencies_satisfied: bool


class EventObservation(BaseModel):
    id: str
    title: str
    event_type: str
    start_minutes: int
    end_minutes: Optional[int] = None
    readiness_metric: Optional[str] = None
    readiness_threshold: Optional[float] = None


class ObservationModel(BaseModel):
    current_time_minutes: int
    steps_remaining: int

    energy: float
    stress: float
    satisfaction: float

    cleanliness: float
    clutter: float
    guest_readiness: float
    kitchen_readiness: float
    laundry_backlog: float

    tasks: list[TaskObservation]
    upcoming_events: list[EventObservation]

    last_action_summary: str
    invalid_action: bool = False


class ActionModel(BaseModel):
    action_type: ACTION_TYPES
    task_id: Optional[str] = None
    minutes: int = Field(default=30, gt=0)

    @field_validator("task_id")
    @classmethod
    def validate_task_id_requirement(cls, value: Optional[str], info) -> Optional[str]:
        action_type = info.data.get("action_type")
        if action_type in {"work_on_task", "defer_task"} and not value:
            raise ValueError(f"task_id is required for action_type='{action_type}'")
        if action_type == "rest" and value is not None:
            raise ValueError("task_id must be None for action_type='rest'")
        return value


class RewardModel(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)


class EnvironmentStateModel(BaseModel):
    scenario_id: str
    day_start_minutes: int = Field(..., ge=0)
    day_end_minutes: int = Field(..., gt=0)
    step_minutes: int = Field(..., gt=0)

    user: UserState
    home: HomeState
    tasks: list[TaskItem]
    events: list[EventItem]

    step_count: int = Field(default=0, ge=0)
    done: bool = False

    @field_validator("day_end_minutes")
    @classmethod
    def day_end_after_start(cls, value: int, info) -> int:
        start = info.data.get("day_start_minutes")
        if start is not None and value <= start:
            raise ValueError("day_end_minutes must be greater than day_start_minutes")
        return value


class ScenarioConfig(BaseModel):
    id: str
    title: str
    description: str

    day_start_minutes: int = Field(default=8 * 60, ge=0)
    day_end_minutes: int = Field(default=22 * 60, gt=0)
    step_minutes: int = Field(default=30, gt=0)

    initial_user: UserState
    initial_home: HomeState
    tasks: list[TaskItem]
    events: list[EventItem]

    @field_validator("day_end_minutes")
    @classmethod
    def validate_day_range(cls, value: int, info) -> int:
        start = info.data.get("day_start_minutes")
        if start is not None and value <= start:
            raise ValueError("day_end_minutes must be greater than day_start_minutes")
        return value
