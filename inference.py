import asyncio
import os
import time
import requests

# ENV CONFIG :-

try:
    API_BASE_URL = os.environ["API_BASE_URL"]
    API_KEY = os.environ["API_KEY"]
except KeyError as e:
    print(f"[FATAL] Missing required env var: {str(e)}", flush=True)
    exit(1)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Build the chat completions URL from API_BASE_URL.
# LiteLLM proxy exposes /v1/chat/completions.
# If the injected URL already ends with /v1 we just append /chat/completions,
# otherwise we add /v1/chat/completions ourselves.
_base = API_BASE_URL.rstrip("/")
if _base.endswith("/v1"):
    CHAT_URL = f"{_base}/chat/completions"
else:
    CHAT_URL = f"{_base}/v1/chat/completions"

print(f"[CONFIG] CHAT_URL={CHAT_URL} MODEL={MODEL_NAME} ENV_URL={ENV_URL}", flush=True)


# LOGGING :-

def log_start():
    print(f"[START] task=customer-support env=custom model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action[:80]} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# WAIT FOR ENV SERVER :-

def wait_for_env(url, retries=12, delay=3):
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


# LLM RESPONSE (raw HTTP — no OpenAI SDK) :-

def get_response(query, history):
    messages = [
        {"role": "system", "content": "You are a helpful and empathetic customer support agent."}
    ]
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["agent"]})
    messages.append({"role": "user", "content": query})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 150
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"[LLM_CALL] POST {CHAT_URL} model={MODEL_NAME}", flush=True)

    resp = requests.post(CHAT_URL, json=payload, headers=headers, timeout=60)

    print(f"[LLM_STATUS] HTTP {resp.status_code}", flush=True)

    if resp.status_code != 200:
        print(f"[LLM_ERROR] body={resp.text[:300]}", flush=True)
        resp.raise_for_status()

    data = resp.json()
    response = data["choices"][0]["message"]["content"].strip()
    print(f"[LLM_OK] {response[:80]}", flush=True)
    return response


async def main():
    log_start()

    if not wait_for_env(ENV_URL):
        print(f"[FATAL] Env server not reachable at {ENV_URL}", flush=True)
        exit(1)

    # RESET
    res = requests.post(f"{ENV_URL}/reset", timeout=20)
    res.raise_for_status()
    result = res.json()

    rewards = []
    step = 0

    for step in range(1, 6):
        obs = result["observation"]
        query = obs["user_query"]
        history = obs["conversation_history"]

        response = get_response(query, history)

        res = requests.post(
            f"{ENV_URL}/step",
            json={"response": response},
            timeout=20
        )
        res.raise_for_status()
        result = res.json()

        reward = result["reward"]
        done = result["done"]

        rewards.append(reward)
        log_step(step, response, reward, done)

        if done:
            break

    success = sum(rewards) > 2.0
    log_end(success, step, rewards)


if __name__ == "__main__":
    asyncio.run(main())
