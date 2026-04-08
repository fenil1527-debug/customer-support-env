from pydantic import BaseModel
from typing import List, Optional, Any, Dict


class HistoryItem(BaseModel):
    user: str
    agent: str


class Observation(BaseModel):
    user_query: str
    conversation_history: List[HistoryItem]
    step: int
    priority: str
    intent: str
    task_id: str
    grader: str


class Action(BaseModel):
    response: str


class State(BaseModel):
    step: int
    intent: Optional[str] = None
    task_id: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}
