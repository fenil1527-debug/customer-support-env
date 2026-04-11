import random
import string
import uuid
from copy import deepcopy

def gen_id(length=5):
    return ''.join(random.choices(string.digits, k=length))

def generate_delivery_task() -> dict:
    order_id = gen_id(5)
    tracking = "TRK" + gen_id(7)
    days = random.randint(3, 12)
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
        "system_db": {order_id: {"status": "shipped", "tracking": tracking}}
    }

def generate_refund_task() -> dict:
    order_id = gen_id(6)
    items = ["mug", "laptop sleeve", "phone case", "water bottle", "speaker"]
    item = random.choice(items)
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
        "system_db": {order_id: {"item": item, "status": "delivered", "eligible": True}}
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
        "system_db": {f"error_{error}": f"{error} Error. Instruct the user to clear their app cache and try again in 5 minutes."}
    }

def get_task() -> dict:
    task_types = [generate_delivery_task, generate_refund_task, generate_technical_task]
    return random.choice(task_types)()

TASKS = [] # Kept for potential legacy references, but get_task generates dynamically
