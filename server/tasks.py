from copy import deepcopy

TASKS = [
    {
        "id": "delivery_issue",
        "query": "My order #5512 hasn’t arrived. Where is it? It's been 5 days.",
        "intent": "delivery",
        "priority": "high",
        "grader": "delivery_grader",
        "system_db": {"5512": {"status": "shipped", "tracking": "TRK992"}}
    },
    {
        "id": "refund_issue",
        "query": "I received a broken mug. Order #8821. I want a refund.",
        "intent": "refund",
        "priority": "high",
        "grader": "refund_grader",
        "system_db": {"8821": {"item": "mug", "status": "delivered", "eligible": True}}
    },
    {
        "id": "technical_issue",
        "query": "The app crashes when I login, it gives error 504.",
        "intent": "technical",
        "priority": "medium",
        "grader": "technical_grader",
        "system_db": {"error_504": "504 Gateway Timeout. Instruct the user to clear their app cache and try again in 5 minutes."}
    },
]

_task_index = 0

def get_task():
    global _task_index
    task = TASKS[_task_index % len(TASKS)]
    _task_index += 1
    return deepcopy(task)
