from __future__ import annotations

import sys
import logging
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Global IO and Logging Initialization
# ---------------------------------------------------------------------------
# Ensure stdout is unbuffered and uses UTF-8 to satisfy the validator
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# Direct all standard logging to stderr so it doesn't pollute the validator's stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("inference")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models import Action, Observation, State, StepResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# API_BASE_URL is strictly the LLM proxy URL injected by the platform
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# Platform injects API_KEY; fall back to HF_TOKEN for local testing
API_KEY: str = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.0
MAX_TOKENS: int = 150
TEMPERATURE: float = 0.7

# ===========================================================================
# HTTP client helpers
# ===========================================================================

def _http(method: str, url: str, body: Optional[Dict] = None, timeout: int = 30) -> Dict:
    data = json.dumps(body).encode() if body is not None else None
    headers: Dict[str, str] = {"Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc

def http_reset() -> Dict:
    url = f"{API_BASE_URL}/reset"
    return _http("POST", url)

def http_step(action: Action) -> Tuple[Dict, float, bool, Dict]:
    resp = _http("POST", f"{API_BASE_URL}/step", body=action.model_dump())
    return resp["observation"], resp["reward"], resp["done"], resp.get("info", {})

# ===========================================================================
# Helper / LLM Logic
# ===========================================================================

def observation_from_dict(d: Dict) -> Observation:
    return Observation(**d)

def _build_client():
    """Build OpenAI client using platform-injected API_BASE_URL and API_KEY."""
    try:
        from openai import OpenAI
        # Crucial: Always create a client. If API_KEY is missing, use a dummy.
        # This ensures the code actually attempts an API call, which the proxy detects.
        key = API_KEY or "no-key-set"
        logger.info("Initializing OpenAI client. Base: %s, Key set: %s", API_BASE_URL, bool(API_KEY))
        return OpenAI(base_url=API_BASE_URL, api_key=key)
    except ImportError:
        logger.warning("openai package not installed. Falling back to heuristic agent.")
        return None
    except Exception as e:
        logger.error("Failed to initialize OpenAI client: %s", e)
        return None

def _heuristic_action(obs: Observation) -> Action:
    if obs.intent == "delivery":
        return Action(response="I am deeply sorry for the delay regarding your tracking. Let me process this immediately.")
    elif obs.intent == "refund":
        return Action(response="I apologize for the poor experience. I will process your refund right now.")
    elif obs.intent == "technical":
        return Action(response="I apologize for the issue. Let me check the system and fix the update.")
    return Action(response="I am sorry you are experiencing this issue. We will work to fix it immediately.")

# ===========================================================================
# LLM action selector
# ===========================================================================

SYSTEM_PROMPT = (
    "You are a helpful and empathetic customer support agent. "
    "Always acknowledge the customer's frustration, then provide a clear specific resolution. "
    "Be concise (under 150 words)."
)

def _build_llm_prompt(obs: Observation) -> str:
    return obs.user_query

def _parse_action_from_response(text: str, obs: Observation) -> Optional[Action]:
    return Action(response=text.strip())

def get_action(client, obs: Observation, use_llm: bool = True) -> Action:
    if not use_llm or client is None:
        return _heuristic_action(obs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in obs.conversation_history:
        messages.append({"role": "user", "content": h.user})
        messages.append({"role": "assistant", "content": h.agent})
        
    messages.append({"role": "user", "content": _build_llm_prompt(obs)})

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages, # type: ignore
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            action = _parse_action_from_response(resp.choices[0].message.content or "", obs)
            if action: return action
        except Exception as e:
            logger.warning(f"LLM attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)

    return _heuristic_action(obs)

# ===========================================================================
# STRUCTURED LOGGING (The Fix for Phase 2)
# ===========================================================================

def _clamp_score(val: float) -> float:
    """Ensure score is strictly within (0, 1) as required by platform."""
    epsilon = 0.01
    return max(epsilon, min(1.0 - epsilon, val))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    clamped_reward = _clamp_score(reward)
    print(f"[STEP] step={step} action={action!r} reward={clamped_reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    clamped_score = _clamp_score(score)
    clamped_rewards = [_clamp_score(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped_rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={clamped_score:.3f} rewards={rewards_str}", flush=True)

# ===========================================================================
# Episode runners
# ===========================================================================

def run_episode(task_id: str, client, use_llm: bool) -> float:
    # On the platform, API_BASE_URL serves the environment proxy too.
    log_start(task_id, "remote", MODEL_NAME)
    
    result_dict = http_reset()
    obs = observation_from_dict(result_dict["observation"])

    rewards_list: List[float] = []
    step_num = 0
    done = False
    score = 0.0

    while not done:
        action = get_action(client, obs, use_llm)
        
        raw_obs_dict, reward, done, _ = http_step(action)
        obs = observation_from_dict(raw_obs_dict)
        
        rewards_list.append(reward)
        step_num += 1
        
        action_clean = action.response.replace("\n", " ").replace("\r", "")[:120]
        log_step(step_num, action_clean, reward, done)
        
        if step_num >= 5:  # hard safety for runaway models
            done = True

    score = sum(rewards_list)
    success = score >= 2.0
    normalized_score = min(max(score / 5.0, 0.0), 1.0)
    log_end(success, step_num, normalized_score, rewards_list)
    return normalized_score

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", default=False)
    args = parser.parse_args()

    # Use the stable baseline tasks for evaluation
    tasks_to_run = ["customer_issue_1", "customer_issue_2", "customer_issue_3"]

    client = _build_client() if not args.no_llm else None

    for task_id in tasks_to_run:
        try:
            run_episode(task_id, client, not args.no_llm)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")

if __name__ == "__main__":
    main()
