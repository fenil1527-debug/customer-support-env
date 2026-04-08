from fastapi import FastAPI

from models import Action, Observation, State, StepResult
# `HistoryItem` is inside `models`, `Observation` and `StepResult` handle their usage.

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

        return StepResult(
            observation=Observation(
                user_query=self.task["query"],
                conversation_history=self.history,  # Wait, wait. Is dict to HistoryItem validated automatically by Pydantic? Let me ensure it's explicitly typed. Let me do self.history where elements are HistoryItems. I will change self.history to contain dicts, FastAPI and Pydantic will auto cast dicts into HistoryItem in the response serialization.
                step=self.step_count,
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
