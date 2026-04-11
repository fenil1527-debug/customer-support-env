def _clamp_strict(score: float) -> float:
    score = round(float(score), 4)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99
    return round(score, 2)


def delivery_grader(history: list, task: dict) -> float:
    score = 0.05
    full_text = " ".join([h["agent"].lower() for h in history])
    
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score)
    order_id = db_keys[0]
    tracking = task["system_db"][order_id]["tracking"].lower()
    
    if "[lookup_order]" in full_text: 
        score += 0.35
        if order_id not in full_text:
            score -= 0.15

    if tracking in full_text:
        score += 0.35
    elif "trk" in full_text: 
        score -= 0.20

    if any(w in full_text for w in ["sorry", "apologize", "understand"]):
        score += 0.15
    if "[message_user]" in full_text:
        score += 0.09
        
    return _clamp_strict(score)


def refund_grader(history: list, task: dict) -> float:
    score = 0.05
    full_text = " ".join([h["agent"].lower() for h in history])
    
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score)
    order_id = db_keys[0]
    
    if "[process_refund]" in full_text:
        score += 0.40
        if order_id not in full_text:
            score -= 0.20
    elif "[lookup_order]" in full_text:
        score += 0.20
        
    if any(w in full_text for w in ["refund", "process", "done", "complete"]):
        score += 0.30
    if any(w in full_text for w in ["sorry", "apologize", "understand"]):
        score += 0.10
    if "[message_user]" in full_text:
        score += 0.10
        
    return _clamp_strict(score)


def technical_grader(history: list, task: dict) -> float:
    score = 0.05
    full_text = " ".join([h["agent"].lower() for h in history])
    
    db_keys = list(task.get("system_db", {}).keys())
    if not db_keys: return _clamp_strict(score)
    error_key = db_keys[0]
    
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
        
    return _clamp_strict(score)


GRADERS = {
    "delivery_grader":  delivery_grader,
    "refund_grader":    refund_grader,
    "technical_grader": technical_grader,
}


def grade_task(history: list, task: dict) -> float:
    grader_name = task.get("grader")
    grader_fn   = GRADERS.get(grader_name)
    
    if grader_fn is None:
        return 0.25

    return grader_fn(history, task)
