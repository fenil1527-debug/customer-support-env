import asyncio
import os
from openai import OpenAI
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy-token")

ENV_URL = os.getenv("ENV_URL")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-token"
)


def log_start():
    print(f"[START] task=customer-support env=custom model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def get_response(query, history):
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "dummy-token"
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

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=150
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return "I’m sorry for the inconvenience. Let me help resolve this issue quickly."


async def main():
    log_start()

    result = requests.post(f"{ENV_URL}/reset").json()
    rewards = []

    for step in range(1, 6):
        obs = result["observation"]
        query = obs["user_query"]
        history = obs["conversation_history"]

        response = get_response(query, history)

        result = requests.post(f"{ENV_URL}/step", json={"response": response}).json()

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
