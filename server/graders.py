def _clamp_strict(score: float) -> float:
    score = round(float(score), 4)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99
    return round(score, 2)

def delivery_grader(history: list, task: dict) -> float:
    score = 0.05
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score)
    order_id = db_keys[0]
    tracking = task["system_db"][order_id]["tracking"].lower()
    
    actions = [h.get("agent", "").lower() for h in history]
    
    has_lookup = False
    has_messaged = False
    hallucinated = False
    revealed_id = False
    hidden_id = task.get("hidden_order_id")
    
    for action in actions:
        if "message_user" in action and ("order" in action or "id" in action):
            revealed_id = True
            
        if "lookup_order" in action:
            if order_id in action:
                has_lookup = True
                if hidden_id == order_id and not revealed_id:
                    hallucinated = True
            else:
                score -= 0.15
        if "message_user" in action:
            has_messaged = True

    if has_lookup:
        score += 0.35
    if hallucinated:
        score -= 0.40  # Heavy penalty for hallucinating the hidden order ID
        
    full_text = " ".join(actions)
    if tracking in full_text:
        score += 0.35
    elif "trk" in full_text:
        score -= 0.20

    if any(w in full_text for w in ["sorry", "apologize", "understand"]):
        score += 0.15
    if has_messaged:
        score += 0.09
    if "ship" in full_text:
        score += 0.05
        
    return _clamp_strict(score)

def refund_grader(history: list, task: dict) -> float:
    score = 0.05
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score)
    order_id = db_keys[0]
    
    actions = [h.get("agent", "").lower() for h in history]
    
    has_refund = False
    has_lookup = False
    hallucinated = False
    revealed_id = False
    hidden_id = task.get("hidden_order_id")

    for action in actions:
        if "message_user" in action and ("order" in action or "id" in action):
            revealed_id = True
            
        if "process_refund" in action:
            if order_id in action:
                has_refund = True
                if hidden_id == order_id and not revealed_id:
                    hallucinated = True
            else:
                score -= 0.20
        elif "lookup_order" in action:
            has_lookup = True
            
    if has_refund:
        score += 0.40
    if has_lookup:
        score += 0.20
    if hallucinated:
        score -= 0.40
        
    full_text = " ".join(actions)
    if any(w in full_text for w in ["refund", "process", "done", "complete"]):
        score += 0.30
    if any(w in full_text for w in ["sorry", "apologize", "understand"]):
        score += 0.10
    if "[message_user]" in full_text:
        score += 0.10
    if "eligible" in str(task.get("system_db", {})).lower():
        score += 0.05
        
    return _clamp_strict(score)

def technical_grader(history: list, task: dict) -> float:
    score = 0.05
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score)
    error_key = db_keys[0]
    
    actions = [h.get("agent", "").lower() for h in history]
    full_text = " ".join(actions)
    
    if "[lookup_kb]" in full_text:
        score += 0.35
        if error_key not in full_text and error_key.replace("error_", "") not in full_text:
            score -= 0.15

    if any(w in full_text for w in ["clear", "cache", "minutes"]):
        score += 0.35
    if any(w in full_text for w in ["apologize", "sorry"]):
        score += 0.10
    if "[message_user]" in full_text:
        score += 0.10
    if "cache" in full_text and "minutes" in full_text:
        score += 0.05
        
    return _clamp_strict(score)

def escalation_grader(history: list, task: dict) -> float:
    score = 0.05
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score)
    
    order_id = [k for k in db_keys if k.isdigit()][0]

    actions = [h.get("agent", "").lower() for h in history]
    
    has_lookup_kb = False
    has_escalated = False
    hallucinated = False
    revealed_id = False
    hidden_id = task.get("hidden_order_id")

    for action in actions:
        if "message_user" in action and ("order" in action or "id" in action):
            revealed_id = True
            
        if "lookup_kb" in action and "policy" in action:
            has_lookup_kb = True
            
        if "escalate_ticket" in action:
            if order_id in action:
                has_escalated = True
                if hidden_id == order_id and not revealed_id:
                    hallucinated = True
            else:
                score -= 0.15
        
        if "process_refund" in action:
            score -= 0.30

    if has_lookup_kb:
        score += 0.25
    if has_escalated:
        score += 0.40
    if hallucinated:
        score -= 0.40

    full_text = " ".join(actions)
    if any(w in full_text for w in ["escalate", "manager", "review", "team"]):
        score += 0.20
    if "premium" in str(task.get("system_db", {})).lower():
        score += 0.05
    if "[message_user]" in full_text:
        score += 0.10
        
    return _clamp_strict(score)

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
