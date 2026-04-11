def _clamp_strict(score: float, patience: float) -> float:
    # Blend the score heavily with patience. If patience is low, reward drops exponentially.
    score = score * (patience ** 1.5)
    score = round(float(score), 4)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99
    return round(score, 2)

def delivery_grader(history: list, task: dict) -> float:
    score = 0.05
    patience = task.get("patience", 1.0)
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score, patience)
    order_id = db_keys[0]
    
    actions = [h.get("agent", "").lower() for h in history]
    full_text = " ".join(actions)
    
    tracking_info = task["system_db"][order_id].get("tracking", "")
    if tracking_info and tracking_info.lower() in full_text:
        score += 0.80
    elif "trk" in full_text:
        score -= 0.20

    if any(w in full_text for w in ["sorry", "apologize", "understand"]):
        score += 0.15
        
    return _clamp_strict(score, patience)

def refund_grader(history: list, task: dict) -> float:
    score = 0.05
    patience = task.get("patience", 1.0)
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score, patience)
    order_id = db_keys[0]
    
    order_obj = task["system_db"][order_id]
    
    # Mathematically check final state
    if order_obj.get("status") == "refunded":
        score += 0.85
    
    actions = [h.get("agent", "").lower() for h in history]
    full_text = " ".join(actions)
    if any(w in full_text for w in ["sorry", "apologize", "understand"]):
        score += 0.10
        
    return _clamp_strict(score, patience)

def technical_grader(history: list, task: dict) -> float:
    score = 0.05
    patience = task.get("patience", 1.0)
    
    actions = [h.get("agent", "").lower() for h in history]
    full_text = " ".join(actions)
    
    if "[lookup_kb]" in full_text:
        score += 0.45

    if any(w in full_text for w in ["clear", "cache", "minutes"]):
        score += 0.40
    if any(w in full_text for w in ["apologize", "sorry"]):
        score += 0.10
        
    return _clamp_strict(score, patience)

def escalation_grader(history: list, task: dict) -> float:
    score = 0.05
    patience = task.get("patience", 1.0)
    db_keys = [k for k in task.get("system_db", {}).keys() if k.isdigit()]
    if not db_keys: return _clamp_strict(score, patience)
    
    order_id = db_keys[0]
    order_obj = task["system_db"][order_id]
    
    if order_obj.get("escalated") == True:
        score += 0.75
        
    actions = [h.get("agent", "").lower() for h in history]
    full_text = " ".join(actions)
    
    if "process_refund" in full_text:
        score -= 0.50 # Penalty for refunding when not eligible
        
    if "[lookup_kb]" in full_text:
        score += 0.10
        
    if any(w in full_text for w in ["apologize", "sorry", "escalate", "manager"]):
        score += 0.10
        
    return _clamp_strict(score, patience)

GRADERS = {
    "delivery_grader":  delivery_grader,
    "refund_grader":    refund_grader,
    "technical_grader": technical_grader,
    "escalation_grader": escalation_grader,
}

def grade_task(history: list, task: dict) -> float:
    grader_name = task.get("grader")
    grader_fn   = GRADERS.get(grader_name)
    
    if grader_fn is None:
        return 0.25

    return grader_fn(history, task)
