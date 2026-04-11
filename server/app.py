from fastapi import FastAPI
import json

from pydantic import BaseModel
from pydantic import Field
from typing import List, Optional, Any, Dict, Literal

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
    patience: float
    budget: float

class Action(BaseModel):
    action_type: Literal["lookup_order", "process_refund", "lookup_kb", "escalate_ticket", "message_user"] = "message_user"
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
    info: Dict[str, Any] = Field(default_factory=dict)

from server.tasks import get_task
from server.graders import grade_task

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "Customer Support OpenEnv is running successfully.",
        "docs": "Append /docs to the URL to interact with the API Swagger UI."
    }

class Env:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 7
        self.task = None
        self.history = []

    def reset(self, task_id: Optional[str] = None):
        self.step_count = 0
        self.task = get_task(task_id)
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
                patience=self.task["patience"],
                budget=self.task["budget"],
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
                self.task["patience"] -= 0.1
                
        elif act_type == "process_refund":
            if target and target in self.task.get("system_db", {}):
                order_data = self.task['system_db'][target]
                if order_data.get("status") == "refunded":
                    system_response = json.dumps({"status": "error", "message": "Already refunded."})
                    self.task["patience"] -= 0.1
                elif "cost" in order_data and self.task["budget"] < order_data["cost"]:
                    system_response = json.dumps({"status": "error", "message": "Insufficient daily budget to process."})
                    self.task["patience"] -= 0.1
                elif order_data.get("eligible") == False:
                    system_response = json.dumps({"status": "error", "message": "Item is not eligible for refund."})
                    self.task["patience"] -= 0.2
                else:
                    cost = order_data.get("cost", 0.0)
                    self.task["budget"] -= cost
                    self.task['system_db'][target]["status"] = "refunded"
                    system_response = json.dumps({"status": "success", "message": f"Refund processed. ${cost} deducted."})
            else:
                system_response = json.dumps({"status": "error", "message": f"Cannot process refund for {target}."})
                self.task["patience"] -= 0.1
                
        elif act_type == "lookup_kb":
            if target and target in self.task.get("system_db", {}):
                system_response = json.dumps({"status": "success", "data": self.task['system_db'][target]})
            else:
                system_response = json.dumps({"status": "error", "message": f"No KB entry found for {target}."})
                self.task["patience"] -= 0.1
                
        elif act_type == "escalate_ticket":
            if target and str(target) in self.task.get("system_db", {}):
                self.task['system_db'][target]["escalated"] = True
                system_response = json.dumps({"status": "success", "message": f"Ticket escalated for {target}."})
            else:
                system_response = json.dumps({"status": "error", "message": f"Cannot escalate: invalid target {target}."})
                self.task["patience"] -= 0.1

        self.history.append({
            "user": self.task["query"],
            "agent": f"[{act_type}] Target={target} | {response}" if target else f"[{act_type}] {response}"
        })

        if act_type == "message_user":
            user_msg = "I have received your message. Please confirm the resolution."
            response_lower = response.lower()
            if "order" in response_lower or "id" in response_lower:
                if self.task.get("hidden_order_id"):
                    user_msg = f"Oh sorry, my order ID is {self.task['hidden_order_id']}."
                    self.task["hidden_order_id"] = None
                else:
                    user_msg = "I believe I already provided my order details."
                    self.task["patience"] -= 0.2  # annoying
            elif "refund" in response_lower and "process" in response_lower:
                user_msg = "Thank you for processing the refund."
            elif "escalat" in response_lower or "manager" in response_lower:
                user_msg = "Thank you, I will wait to hear from the manager."
            elif "cache" in response_lower:
                user_msg = "I cleared my cache, but let's assume it works now."
            else:
                self.task["patience"] -= 0.05
            self.task["query"] = user_msg

        self.task["patience"] = max(0.0, round(self.task["patience"], 2))
        self.task["budget"] = round(self.task["budget"], 2)

        score = grade_task(self.history, self.task)
        
        # Patience burnout ends the episode in failure
        if self.task["patience"] <= 0.0:
            score = 0.0
            done = True
        else:
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
                patience=self.task["patience"],
                budget=self.task["budget"],
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
def reset(task_id: Optional[str] = None):
    return env.reset(task_id)

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
