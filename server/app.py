from fastapi import FastAPI
import json

from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class HistoryItem(BaseModel):
    user: str
    agent: str

class Observation(BaseModel):
    user_query: str
    conversation_history: List[HistoryItem]
    step: int
    system_response: Optional[str] = None
    priority: str
    intent: str
    task_id: str
    grader: str

class Action(BaseModel):
    action_type: str = "message_user"
    target: Optional[str] = None
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

from server.tasks import get_task
from server.graders import grade_task

app = FastAPI()


class Env:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5
        self.task = None
        self.history = []

    def reset(self):
        self.step_count = 0
        self.task = get_task()
        self.history = []

        return StepResult(
            observation=Observation(
                user_query=self.task["query"],
                conversation_history=[],
                step=0,
                system_response=None,
                priority=self.task["priority"],
                intent=self.task["intent"],
                task_id=self.task["id"],
                grader=self.task["grader"],
            ),
            reward=0.01,
            done=False,
        )

    def step(self, action):
        self.step_count += 1
        act_type = action.get("action_type", "message_user")
        target = action.get("target")
        response = action.get("response", "")

        system_response = None
        reward = 0.0

        if act_type == "lookup_order":
            if target and target in self.task.get("system_db", {}):
                system_response = json.dumps({"status": "success", "data": self.task['system_db'][target]})
            else:
                system_response = json.dumps({"status": "error", "message": f"Order {target} not found."})
                
        elif act_type == "process_refund":
            if target and target in self.task.get("system_db", {}):
                system_response = json.dumps({"status": "success", "message": f"Refund processed for {target}."})
            else:
                system_response = json.dumps({"status": "error", "message": f"Cannot process refund for {target}."})
                
        elif act_type == "lookup_kb":
            if target and target in self.task.get("system_db", {}):
                system_response = json.dumps({"status": "success", "data": self.task['system_db'][target]})
            else:
                system_response = json.dumps({"status": "error", "message": f"No KB entry found for {target}."})

        self.history.append({
            "user": self.task["query"],
            "agent": f"[{act_type}] Target={target} | {response}" if target else f"[{act_type}] {response}"
        })

        score = grade_task(self.history, self.task)

        if act_type == "message_user":
            self.task["query"] = "I have received your message. Please confirm the resolution."

        done = self.step_count >= self.max_steps or score > 0.85

        return StepResult(
            observation=Observation(
                user_query=self.task["query"],
                conversation_history=self.history,
                step=self.step_count,
                system_response=system_response,
                priority=self.task["priority"],
                intent=self.task["intent"],
                task_id=self.task["id"],
                grader=self.task["grader"],
            ),
            reward=score,
            done=done,
        )

    def state(self) -> State:
        if self.task is None:
            return State(step=self.step_count, intent=None, task_id=None)
        return State(
            step=self.step_count,
            intent=self.task["intent"],
            task_id=self.task["id"],
        )


env = Env()


@app.post("/reset", response_model=StepResult)
def reset():
    return env.reset()


@app.post("/step", response_model=StepResult)
def step(action: Action):
    return env.step(action.model_dump())


@app.get("/state", response_model=State)
def state():
    return env.state()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
