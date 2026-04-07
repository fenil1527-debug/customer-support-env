from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

@app.get("/")
def home():
    return {"status": "env running"}

# ===== TASKS =====
TASKS = [
    {
        "query": "My order #1234 hasn’t arrived and it’s been 5 days.",
        "intent": "delivery",
        "priority": "high",
        "expected_keywords": ["sorry", "track", "delay"]
    },
    {
        "query": "I want a refund for a damaged product.",
        "intent": "refund",
        "priority": "high",
        "expected_keywords": ["refund", "apologize", "process"]
    },
    {
        "query": "The app crashes when I open it.",
        "intent": "technical",
        "priority": "medium",
        "expected_keywords": ["update", "fix", "issue"]
    }
]

# ===== ENV =====
class Env:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5
        self.task = None
        self.history = []

    def reset(self):
        self.step_count = 0
        self.task = random.choice([
            {
                "query": "My order #1234 hasn’t arrived and it’s been 5 days.",
                "intent": "delivery",
                "priority": "high",
                "complexity": "medium",
                "expected_keywords": ["sorry", "track", "delay"]
            },
            {
                "query": "I want a refund for a damaged product.",
                "intent": "refund",
                "priority": "high",
                "complexity": "easy",
                "expected_keywords": ["refund", "apologize", "process"]
            },
            {
                "query": "The app crashes when I open it.",
                "intent": "technical",
                "priority": "medium",
                "complexity": "hard",
                "expected_keywords": ["update", "fix", "issue"]
            }
        ])
        self.history = []

        return {
            "observation": {
                "user_query": self.task["query"],
                "conversation_history": [],
                "step": 0,
                "priority": self.task["priority"],
                "intent": self.task["intent"]
            },
            "reward": 0,
            "done": False
        }

    def grade_response(self, response):
        response_lower = response.lower()
        score = 0.0

        # 1. Empathy
        if any(w in response_lower for w in ["sorry", "apologize", "understand"]):
            score += 0.25

        # 2. Actionability
        if any(w in response_lower for w in ["track", "refund", "update", "fix", "check"]):
            score += 0.35

        # 3. Task relevance
        for kw in self.task["expected_keywords"]:
            if kw in response_lower:
                score += 0.15

        # 4. Clarity
        if 30 < len(response) < 300:
            score += 0.15

        # 5. Penalties
        if len(response) < 10:
            score -= 0.3
        if "idk" in response_lower or "don't know" in response_lower:
            score -= 0.5

        # 6. Wrong intent penalty
        if "refund" in response_lower and self.task["intent"] != "refund":
            score -= 0.3

        # 7. Priority boost
        if self.task["priority"] == "high":
            score *= 1.2

        # 8. Complexity boost
        if self.task["complexity"] == "hard":
            score *= 1.3

        return max(0.0, min(score, 1.0))

    def step(self, action):
        self.step_count += 1
        response = action["response"]

        score = self.grade_response(response)

        # Conversation memory
        self.history.append({
            "user": self.task["query"],
            "agent": response
        })

        # Dynamic query evolution
        if self.step_count > 1:
            self.task["query"] = "User is still waiting. Please provide a better resolution."

        done = self.step_count >= self.max_steps or score > 0.85

        return {
            "observation": {
                "user_query": self.task["query"],
                "conversation_history": self.history,
                "step": self.step_count,
                "priority": self.task["priority"],
                "intent": self.task["intent"]
            },
            "reward": round(score, 2),
            "done": done
        }

    def state(self):
        return {
            "step": self.step_count,
            "intent": self.task["intent"]
        }


env = Env()

# ===== API MODEL =====
class Action(BaseModel):
    response: str

# ===== ENDPOINTS =====

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    return env.step(action.dict())

@app.get("/state")
def state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()