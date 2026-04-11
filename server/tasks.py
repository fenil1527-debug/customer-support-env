import random
import string
import uuid
from typing import Callable, Dict, List, Optional

def gen_id(length=5):
    return ''.join(random.choices(string.digits, k=length))

def generate_delivery_task() -> dict:
    order_id = gen_id(5)
    tracking = "TRK" + gen_id(7)
    days = random.randint(3, 12)
    hide_id = random.choice([True, False])
    if hide_id:
        templates = [
            f"My order hasn't arrived. Where is it? It's been {days} days.",
            f"Can you check on my order? Usually it's faster but it's been {days} days.",
            f"I need the tracking for my order. Is it lost?"
        ]
    else:
        templates = [
            f"My order #{order_id} hasn't arrived. Where is it? It's been {days} days.",
            f"Can you check on order {order_id}? Usually it's faster but it's been {days} days.",
            f"I need the tracking for order_id {order_id}. Is it lost?"
        ]
    return {
        "id": f"delivery_issue_{uuid.uuid4().hex[:6]}",
        "query": random.choice(templates),
        "intent": "delivery",
        "priority": "high",
        "grader": "delivery_grader",
        "hidden_order_id": order_id if hide_id else None,
        "patience": 1.0,
        "budget": 500.0,
        "system_db": {order_id: {"status": "shipped", "tracking": tracking}}
    }

def generate_refund_task() -> dict:
    order_id = gen_id(6)
    items = ["mug", "laptop sleeve", "phone case", "water bottle", "speaker"]
    item = random.choice(items)
    hide_id = random.choice([True, False])
    cost = round(random.uniform(20.0, 150.0), 2)
    if hide_id:
        templates = [
            f"I received a broken {item}. I want a refund.",
            f"Refund me for the {item}, it arrived damaged.",
            f"My {item} is defective. Is a refund possible?"
        ]
    else:
        templates = [
            f"I received a broken {item}. Order #{order_id}. I want a refund.",
            f"Refund me for the {item} from order {order_id}, it arrived damaged.",
            f"Order {order_id} has a defective {item}. Is a refund possible?"
        ]
    return {
        "id": f"refund_issue_{uuid.uuid4().hex[:6]}",
        "query": random.choice(templates),
        "intent": "refund",
        "priority": "high",
        "grader": "refund_grader",
        "hidden_order_id": order_id if hide_id else None,
        "patience": 1.0,
        "budget": 500.0,
        "system_db": {order_id: {"item": item, "status": "delivered", "eligible": True, "cost": cost}}
    }

def generate_technical_task() -> dict:
    error_codes = ["504", "500", "403", "401"]
    error = random.choice(error_codes)
    templates = [
        f"The app crashes when I login, it gives error {error}.",
        f"I'm seeing an error {error} on the dashboard page.",
        f"Help! I keep getting {error} error when checking out."
    ]
    return {
        "id": f"technical_issue_{uuid.uuid4().hex[:6]}",
        "query": random.choice(templates),
        "intent": "technical",
        "priority": "medium",
        "grader": "technical_grader",
        "patience": 1.0,
        "budget": 500.0,
        "system_db": {f"error_{error}": {"type": "kb_entry", "content": f"{error} Error. Instruct the user to clear their app cache and try again in 5 minutes."}}
    }

def generate_escalation_task() -> dict:
    order_id = gen_id(6)
    hide_id = random.choice([True, False])
    if hide_id:
        templates = [
            f"My premium laptop broke after 40 days. Can I get a refund?",
            f"I need to return my monitor. It's been 45 days, but I have a premium warranty."
        ]
    else:
        templates = [
            f"Order #{order_id} was a premium laptop that broke after 40 days. Can I get a refund?",
            f"I need to return my monitor from order {order_id}. It's been 45 days, but I have a premium warranty."
        ]
    return {
        "id": f"escalation_issue_{uuid.uuid4().hex[:6]}",
        "query": random.choice(templates),
        "intent": "escalation",
        "priority": "high",
        "grader": "escalation_grader",
        "hidden_order_id": order_id if hide_id else None,
        "patience": 1.0,
        "budget": 500.0,
        "system_db": {
            "policy_premium_return": {"type": "kb_entry", "content": "Premium returns >30 days require manager approval. Do NOT process_refund. Use escalate_ticket with the order_id."},
            order_id: {"item": "electronics", "status": "delivered", "days_since_purchase": 40 + random.randint(0, 10), "eligible": False, "requires_escalation": True}
        }
    }

_TASK_GENERATORS: Dict[str, Callable[[], dict]] = {
    "delivery": generate_delivery_task,
    "refund": generate_refund_task,
    "technical": generate_technical_task,
    "escalation": generate_escalation_task,
}

def get_task(task_id: Optional[str] = None) -> dict:
    if task_id:
        family = task_id.lower().strip()
        if family in _TASK_GENERATORS:
            return _TASK_GENERATORS[family]()
        for key, generator in _TASK_GENERATORS.items():
            if family.startswith(key):
                return generator()

    task_types = list(_TASK_GENERATORS.values())
    return random.choice(task_types)()
