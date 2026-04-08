from copy import deepcopy

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

_task_index = 0

def get_task():
    global _task_index
    task = TASKS[_task_index % len(TASKS)]
    _task_index += 1
    return deepcopy(task)
