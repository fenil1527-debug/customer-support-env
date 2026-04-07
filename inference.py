import asyncio
import os
from openai import OpenAI
import requests

# ENV CONFIG :-

try:
    API_BASE_URL = os.environ["API_BASE_URL"]
    API_KEY = os.environ["API_KEY"]
except KeyError as e:
    print(f"[FATAL] Missing required env var: {str(e)}", flush=True)
    exit(1)

# Normalize base URL: openai SDK v1.x expects the URL to end with /v1
if not API_BASE_URL.rstrip("/").endswith("/v1"):
    API_BASE_URL = API_BASE_URL.rstrip("/") + "/v1"

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

print(f"[CONFIG] API_BASE_URL={API_BASE_URL} MODEL_NAME={MODEL_NAME} ENV_URL={ENV_URL}", flush=True)


# LOGGING :-

def log_start():
    print(f"[START] task=customer-support env=custom model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# LLM RESPONSE :-

def get_response(query, history):
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and empathetic customer support agent."
        }
    ]

    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["agent"]})

    messages.append({"role": "user", "content": query})

    print(f"[LLM_CALL] Sending request to {API_BASE_URL} with model={MODEL_NAME}", flush=True)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=150
    )

    response = completion.choices[0].message.content.strip()
    print(f"[LLM_RESPONSE] {response[:80]}...", flush=True)
    return response


async def main():
    log_start()

    try:
        # RESET
        res = requests.post(f"{ENV_URL}/reset", timeout=20)
        res.raise_for_status()
        result = res.json()

        rewards = []

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

    except Exception as e:
        print(f"[FATAL] {str(e)}", flush=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
