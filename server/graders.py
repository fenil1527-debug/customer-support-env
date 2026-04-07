def _clamp_strict(score: float) -> float:
    score = float(score)
    if score <= 0:
        score = 0.01
    elif score >= 1:
        score = 0.99
    return round(score, 2)


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
    grader_name = task.get("grader")
    grader_fn = GRADERS.get(grader_name)

    print(f"[GRADER] Using {grader_name}", flush=True)

    if grader_fn is None:
        return 0.25

    return grader_fn(response, task)
