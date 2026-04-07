def _clamp_strict(score: float) -> float:
    return round(max(0.01, min(float(score), 0.99)), 2)


def delivery_grader(response: str, task: dict) -> float:
    text = response.lower()
    score = 0.18

    if any(w in text for w in ["sorry", "apologize", "understand"]):
        score += 0.25
    if any(w in text for w in ["track", "tracking", "delay", "delayed", "check"]):
        score += 0.25
    if any(kw in text for kw in task["expected_keywords"]):
        score += 0.20
    if 30 < len(response) < 300:
        score += 0.12
    if "refund" in text:
        score -= 0.05

    return _clamp_strict(score)


def refund_grader(response: str, task: dict) -> float:
    text = response.lower()
    score = 0.20

    if any(w in text for w in ["sorry", "apologize", "understand"]):
        score += 0.22
    if any(w in text for w in ["refund", "return", "process", "replace"]):
        score += 0.28
    if any(kw in text for kw in task["expected_keywords"]):
        score += 0.18
    if 30 < len(response) < 300:
        score += 0.12
    if "track" in text and "refund" not in text:
        score -= 0.05

    return _clamp_strict(score)


def technical_grader(response: str, task: dict) -> float:
    text = response.lower()
    score = 0.16

    if any(w in text for w in ["sorry", "apologize", "understand"]):
        score += 0.18
    if any(w in text for w in ["update", "fix", "issue", "restart", "clear cache"]):
        score += 0.32
    if any(kw in text for kw in task["expected_keywords"]):
        score += 0.18
    if 35 < len(response) < 320:
        score += 0.10
    if "refund" in text:
        score -= 0.08

    return _clamp_strict(score)


GRADERS = {
    "delivery_grader": delivery_grader,
    "refund_grader": refund_grader,
    "technical_grader": technical_grader,
}

def grade_task(response: str, task: dict) -> float:
    grader_fn = GRADERS.get(task.get("grader"))
    if grader_fn is None:
        return 0.25
    return _clamp_strict(grader_fn(response, task))
