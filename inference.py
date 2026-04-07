"""
inference.py — customer-support-env
Follows the mandatory STDOUT format:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import os
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────────

try:
    API_BASE_URL = os.environ["API_BASE_URL"]
    API_KEY      = os.environ["API_KEY"]
except KeyError as e:
    print(f"[FATAL] Missing required env var: {e}", flush=True)
    exit(1)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL    = os.getenv("ENV_URL", "http://localhost:7860")

TASK_NAME  = "customer-support"
BENCHMARK  = "customer-support-env"
MAX_STEPS  = 5
SUCCESS_THRESHOLD = 2.0   # sum of rewards needed to count as success

print(f"[CONFIG] API_BASE_URL={API_BASE_URL} MODEL={MODEL_NAME} ENV_URL={ENV_URL}", flush=True)

# ── LOGGING ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Collapse newlines so the whole action stays on one line
    action_single = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_single} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── WAIT FOR ENV SERVER ───────────────────────────────────────────────────────

def wait_for_env(url: str, retries: int = 15, delay: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{url}/", timeout=5)
            if r.status_code == 200:
                print(f"[ENV] Ready after {attempt} attempt(s)", flush=True)
                return True
        except Exception as e:
            print(f"[ENV] Attempt {attempt}/{retries}: {e}", flush=True)
        time.sleep(delay)
    return False

# ── LLM CALL (OpenAI SDK — mandatory per spec) ────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful and empathetic customer support agent. "
    "Give clear, actionable, concise responses. "
    "Always acknowledge the customer's frustration, then provide a specific resolution."
)

def get_response(client: OpenAI, query: str, history: list, intent: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for h in history:
        messages.append({"role": "user",      "content": h["user"]})
        messages.append({"role": "assistant", "content": h["agent"]})

    messages.append({"role": "user", "content": query})

    print(f"[LLM_CALL] model={MODEL_NAME} intent={intent} history_len={len(history)}", flush=True)

    # Let exceptions propagate — validator needs to see real errors
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )

    text = (completion.choices[0].message.content or "").strip()
    print(f"[LLM_OK] {text[:80]}", flush=True)
    return text if text else "I apologize for the inconvenience. Let me help you resolve this right away."

# ── MAIN ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Wait for the FastAPI env container to be ready
    if not wait_for_env(ENV_URL):
        print(f"[FATAL] Env server not reachable at {ENV_URL}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        exit(1)

    try:
        # RESET
        res = requests.post(f"{ENV_URL}/reset", timeout=20)
        res.raise_for_status()
        result = res.json()

        for step in range(1, MAX_STEPS + 1):
            obs     = result["observation"]
            query   = obs["user_query"]
            history = obs["conversation_history"]
            intent  = obs.get("intent", "general")

            response = get_response(client, query, history, intent)

            step_res = requests.post(
                f"{ENV_URL}/step",
                json={"response": response},
                timeout=20,
            )
            step_res.raise_for_status()
            result = step_res.json()

            reward = float(result["reward"])
            done   = bool(result["done"])

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=response, reward=reward, done=done, error=None)

            if done:
                break

        total  = sum(rewards)
        # Normalize: max possible = MAX_STEPS * 1.0 = 5.0, clamp to (0,1)
        score  = min(max(total / MAX_STEPS, 0.001), 0.999)
        success = total >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", flush=True)
        raise

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
