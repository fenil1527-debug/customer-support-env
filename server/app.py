from fastapi import FastAPI
from pydantic import BaseModel

from server.tasks import get_task
from server.graders import grade_task

app = FastAPI()


@app.get("/")
def home():
    return {"status": "env running"}


class Action(BaseModel):
    response: str


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
            "reward": 0.01,
            "done": False,
        }

    def step(self, action):
        self.step_count += 1
        response = action["response"]

        score = grade_task(response, self.task)

        self.history.append(
            {
                "user": self.task["query"],
                "agent": response,
            }
        )

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
