from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")

TASK_NAME = "customer-support"
BENCHMARK = "customer-support-env"
MAX_STEPS = 5
SUCCESS_THRESHOLD = 2.0

# ===========================================================================
# Client Initialization
# ===========================================================================

def _build_client():
    """Build OpenAI client. Always creates one even with dummy key."""
    # Crucial: Always create a client. If API_KEY is missing, use a dummy.
    key = API_KEY or "no-key-set"
    logger.info("Initializing OpenAI client. Base: %s, Key set: %s", API_BASE_URL, bool(API_KEY))
    return OpenAI(base_url=API_BASE_URL, api_key=key)

# ===========================================================================
# LLM Logic / Fallbacks
# ===========================================================================

def _heuristic_action(query: str, intent: str) -> str:
    """Fallback logic if the LLM fails or is disabled."""
    if intent == "delivery":
        return "I am deeply sorry for the delay regarding your tracking. Let me process this immediately."
    elif intent == "refund":
        return "I apologize for the poor experience. I will process your refund right now."
    elif intent == "technical":
        return "I apologize for the issue. Let me check the system and fix the update."
    return "I am sorry you are experiencing this issue. We will work to fix it immediately."

SYSTEM_PROMPT = (
    "You are a helpful and empathetic customer support agent. "
    "Always acknowledge the customer's frustration, then provide a clear specific resolution. "
    "Be concise (under 150 words)."
)

def get_action(client: Optional[OpenAI], query: str, history: list, intent: str, use_llm: bool = True) -> str:
    if not use_llm or client is None:
        return _heuristic_action(query, intent)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": "user", "content": h.get("user", "")})
        messages.append({"role": "assistant", "content": h.get("agent", "")})
        
    messages.append({"role": "user", "content": query})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages, # type: ignore
                max_tokens=150,
                temperature=0.7,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("LLM attempt %d failed: %s", attempt + 1, e)
            time.sleep(1.0)

    return _heuristic_action(query, intent)

# ===========================================================================
# STRUCTURED LOGGING
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
    
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(f"[STEP] step={step} action={action_clean!r} reward={clamped_reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    clamped_score = _clamp_score(score)
    clamped_rewards = [_clamp_score(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped_rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={clamped_score:.3f} rewards={rewards_str}", flush=True)

# ===========================================================================
# Episode Runner
# ===========================================================================

def wait_for_env(url: str, retries: int = 5, delay: int = 1) -> bool:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{url}/", timeout=5)
            if r.status_code == 200:
                logger.info("Env Ready after %d attempts", attempt)
                return True
        except Exception as exc:
            logger.warning("Env wait attempt %d/%d: %s", attempt, retries, exc)
        time.sleep(delay)
    return False

def run_episode(client: Optional[OpenAI], use_llm: bool) -> None:
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    if not wait_for_env(ENV_URL):
        logger.error("Environment server not reachable at %s. Did you start the server?", ENV_URL)
        log_end(False, 0, 0.0, [])
        return
        
    try:
        res = requests.post(f"{ENV_URL}/reset", timeout=20)
        res.raise_for_status()
        result = res.json()
        
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        for step in range(1, MAX_STEPS + 1):
            obs = result.get("observation", {})
            query = obs.get("user_query", "")
            history = obs.get("conversation_history", [])
            intent = obs.get("intent", "general")
            
            response = get_action(client, query, history, intent, use_llm)
            
            step_res = requests.post(f"{ENV_URL}/step", json={"response": response}, timeout=20)
            step_res.raise_for_status()
            result = step_res.json()
            
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=response, reward=reward, done=done)
            
            if done:
                break
                
        total = sum(rewards)
        score = min(max(total / MAX_STEPS, 0.0), 1.0)
        success = total >= SUCCESS_THRESHOLD
        
    except Exception as e:
        logger.error("Episode failed: %s", e)
        log_end(False, 0, 0.0, [])
        return
        
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", default=False, help="Disable LLM and use a heuristic policy")
    args = parser.parse_args()

    client = _build_client() if not args.no_llm else None

    # Customer support env iterates sequentially over tasks, so a single run_episode triggers one
    # To evaluate multiple, we could loop, but for baselines we run 3 times
    for task_idx in range(3):
        logger.info(f"--- Starting task loop {task_idx+1}/3 ---")
        run_episode(client, not args.no_llm)

if __name__ == "__main__":
    main()
