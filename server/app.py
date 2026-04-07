from fastapi import FastAPI
from pydantic import BaseModel
from copy import deepcopy
import random

app = FastAPI()


@app.get("/")
def home():
    return {"status": "env running"}


def _clamp_score(score: float) -> float:
    # Validator requires strictly between 0 and 1
    return round(max(0.01, min(float(score), 0.99)), 2)


# Three explicit tasks, each with its own grader metadata
TASKS = [
    {
        "id": "delivery_issue",
        "query": "My order #1234 hasn’t arrived and it’s been 5 days.",
        "intent": "delivery",
        "priority": "high",
        "complexity": "medium",
        "expected_keywords": ["sorry", "track", "delay"],
        "grader": "delivery_grader",
    },
    {
        "id": "refund_issue",
        "query": "I want a refund for a damaged product.",
        "intent": "refund",
        "priority": "high",
        "complexity": "easy",
        "expected_keywords": ["refund", "apologize", "process"],
        "grader": "refund_grader",
    },
    {
        "id": "technical_issue",
        "query": "The app crashes when I open it.",
        "intent": "technical",
        "priority": "medium",
        "complexity": "hard",
        "expected_keywords": ["update", "fix", "issue"],
        "grader": "technical_grader",
    },
]


def delivery_grader(response: str, task: dict) -> float:
    text = response.lower()
    score = 0.18

    if any(w in text for w in ["sorry", "apologize", "understand"]):
        score += 0.25
    if any(w in text for w in ["track", "tracking", "delay", "delayed", "check"]):
        score += 0.25
    if any(kw in text for kw in task["expected_keywords"]):
        score += 0.20
    if 30 < len(response) < 300:
        score += 0.12
    if "refund" in text:
        score -= 0.05

    return _clamp_score(score)


def refund_grader(response: str, task: dict) -> float:
    text = response.lower()
    score = 0.20

    if any(w in text for w in ["sorry", "apologize", "understand"]):
        score += 0.22
    if any(w in text for w in ["refund", "return", "process", "replace"]):
        score += 0.28
    if any(kw in text for kw in task["expected_keywords"]):
        score += 0.18
    if 30 < len(response) < 300:
        score += 0.12
    if "track" in text and "refund" not in text:
        score -= 0.05

    return _clamp_score(score)


def technical_grader(response: str, task: dict) -> float:
    text = response.lower()
    score = 0.16

    if any(w in text for w in ["sorry", "apologize", "understand"]):
        score += 0.18
    if any(w in text for w in ["update", "fix", "issue", "restart", "clear cache"]):
        score += 0.32
    if any(kw in text for kw in task["expected_keywords"]):
        score += 0.18
    if 35 < len(response) < 320:
        score += 0.10
    if "refund" in text:
        score -= 0.08

    return _clamp_score(score)


GRADERS = {
    "delivery_grader": delivery_grader,
    "refund_grader": refund_grader,
    "technical_grader": technical_grader,
}


class Env:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5
        self.task = None
        self.history = []

    def reset(self):
        self.step_count = 0
        self.task = deepcopy(random.choice(TASKS))
        self.history = []

        return {
            "observation": {
                "user_query": self.task["query"],
                "conversation_history": [],
                "step": 0,
                "priority": self.task["priority"],
                "intent": self.task["intent"],
                "task_id": self.task["id"],
                "grader": self.task["grader"],
            },
            "reward": 0.0,
            "done": False,
        }

    def grade_response(self, response: str) -> float:
        grader_name = self.task.get("grader", "")
        grader_fn = GRADERS.get(grader_name)

        if grader_fn is None:
            # Safe fallback still kept strictly inside (0, 1)
            return 0.25

        score = grader_fn(response, self.task)

        # Keep final reward strictly between 0 and 1
        return _clamp_score(score)

    def step(self, action):
        self.step_count += 1
        response = action["response"]

        score = self.grade_response(response)

        self.history.append(
            {
                "user": self.task["query"],
                "agent": response,
            }
        )

        # Make follow-up prompt evolve after the first turn
        if self.step_count > 1:
            self.task["query"] = "User is still waiting. Please provide a better resolution."

        done = self.step_count >= self.max_steps or score > 0.85

        return {
            "observation": {
                "user_query": self.task["query"],
                "conversation_history": self.history,
                "step": self.step_count,
                "priority": self.task["priority"],
                "intent": self.task["intent"],
                "task_id": self.task["id"],
                "grader": self.task["grader"],
            },
            "reward": score,
            "done": done,
        }

    def state(self):
        if self.task is None:
            return {"step": self.step_count, "intent": None, "task_id": None}
        return {
            "step": self.step_count,
            "intent": self.task["intent"],
            "task_id": self.task["id"],
        }


env = Env()


class Action(BaseModel):
    response: str


@app.post("/reset")
def reset():
    return env.reset()


@app.post("/step")
def step(action: Action):
    return env.step(action.model_dump())


@app.get("/state")
def state():
    return env.state()


def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
