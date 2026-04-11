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


sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("inference")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server.app import Env as CustomerSupportEnv
from server.app import Observation, Action


API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
USE_LOCAL_ENV: bool = os.getenv("USE_LOCAL_ENV", "1") == "1"

MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.0
MAX_TOKENS: int = 150
TEMPERATURE: float = 0.7



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

def http_reset(task_id: str) -> Dict:
    url = f"{API_BASE_URL}/reset?task_id={urllib.parse.quote(task_id)}"
    return _http("POST", url)

def http_step(action: Action) -> Tuple[Dict, float, bool, Dict]:
    resp = _http("POST", f"{API_BASE_URL}/step", body=action.model_dump())
    return resp["observation"], resp["reward"], resp["done"], resp.get("info", {})



def observation_from_dict(d: Dict) -> Observation:
    return Observation(**d)

def _build_client():
    from openai import OpenAI
    key = API_KEY or "no-key-set"
    logger.info("Initializing OpenAI client. Base: %s, Key set: %s", API_BASE_URL, bool(API_KEY))
    try:

        return OpenAI(base_url=API_BASE_URL, api_key=key)
    except Exception as e:
        logger.warning(f"OpenAI package instantiation crashed: {e}. Falling back to pure HTTP proxy bypass.")
        return "USE_HTTP"

def _http_llm_call(messages: List[Dict[str, str]]) -> str:

    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY or 'dummy'}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


def _heuristic_action(obs: Observation) -> Action:
    history_len = len(obs.conversation_history)
    
    if obs.intent == "delivery":
        if history_len == 0:
            order_id = "0000"
            m = re.search(r"\d{5,6}", obs.user_query)
            if m: order_id = m.group(0)
            return Action(action_type="lookup_order", target=order_id, response="Let me check.")
        else:
            tracking = "unknown"
            if obs.system_response:
                m = re.search(r"TRK\d+", obs.system_response, re.IGNORECASE)
                if m: tracking = m.group(0)
                else:
                    m = re.search(r'"tracking":\s*"([^"]+)"', obs.system_response, re.IGNORECASE)
                    if m: tracking = m.group(1)
            return Action(action_type="message_user", response=f"I apologize, your tracking is {tracking}. It has shipped.")
            
    elif obs.intent == "refund":
        if history_len == 0:
            order_id = "0000"
            m = re.search(r"\d{5,6}", obs.user_query)
            if m: order_id = m.group(0)
            return Action(action_type="process_refund", target=order_id, response="Processing refund.")
        else:
            return Action(action_type="message_user", response="I apologize, I have processed your refund.")
            
    elif obs.intent == "technical":
        if history_len == 0:
            error_code = "504"
            m = re.search(r"error\s+(\d+)", obs.user_query, re.IGNORECASE)
            if m: error_code = m.group(1)
            return Action(action_type="lookup_kb", target=f"error_{error_code}", response="Checking KB.")
        else:
            return Action(action_type="message_user", response="I apologize, please clear your cache and try in 5 minutes.")

    return Action(action_type="message_user", response="I am sorry you are experiencing this issue. We will work to fix it immediately.")

SYSTEM_PROMPT = """You are an advanced Customer Support AI. You have access to internal systems.
You MUST output a raw JSON object string to act. Schema:
{
  "action_type": "lookup_order" | "process_refund" | "lookup_kb" | "message_user",
  "target": "order_id or kb_topic",
  "response": "message to user or internal thought"
}
Available Actions:
- 'lookup_order': Target is order_id. Returns status and tracking in JSON.
- 'process_refund': Target is order_id. Processes a refund.
- 'lookup_kb': Target is kb_topic (e.g., error_504). Returns knowledge base entry in JSON.
- 'message_user': No target. Final action to communicate with the user.
Always respond with pure JSON."""

def _build_llm_prompt(obs: Observation) -> str:
    return obs.user_query

def _parse_action_from_response(text: str, obs: Observation) -> Optional[Action]:
    try:
        json_str = text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
            
        data = json.loads(json_str.strip())
        return Action(**data)
    except Exception as e:
        logger.warning(f"Failed to parse Action JSON. Raw: {text}")
        return Action(action_type="message_user", response=text.strip())

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
            if client == "USE_HTTP":
                text = _http_llm_call(messages)
            else:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                text = resp.choices[0].message.content or ""
            
            action = _parse_action_from_response(text, obs)
            if action: return action
        except Exception as e:
            logger.warning(f"LLM proxy attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)

    return _heuristic_action(obs)



def _clamp_score(val: float) -> float:
    epsilon = 0.01
    return max(epsilon, min(1.0 - epsilon, val))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action!r} reward={_clamp_score(reward):.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{_clamp_score(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={_clamp_score(score):.3f} rewards={rewards_str}", flush=True)



def run_episode(task_id: str, client, use_llm: bool, use_local: bool) -> float:
    env_display = "local" if use_local else "remote"
    log_start(task_id, env_display, MODEL_NAME)

    if use_local:
        env = CustomerSupportEnv()
        result_obj = env.reset()
        obs = result_obj.observation
    else:
        result_dict = http_reset(task_id)
        obs = observation_from_dict(result_dict["observation"])
        env = CustomerSupportEnv()

    rewards_list: List[float] = []
    step_num = 0
    done = False

    while not done:
        action = get_action(client, obs, use_llm)

        if use_local:
            step_result = env.step(action.model_dump())
            obs = step_result.observation
            reward = step_result.reward
            done = step_result.done
        else:
            raw_obs_dict, reward, done, _ = http_step(action)
            obs = observation_from_dict(raw_obs_dict)
            try: env.step(action.model_dump())
            except: pass

        rewards_list.append(reward)
        step_num += 1

        action_clean = action.response.replace("\n", " ").replace("\r", "")[:120]
        log_step(step_num, action_clean, reward, done)

        if step_num >= 5: done = True


    normalized_score = max(rewards_list) if rewards_list else 0.0
    success = normalized_score >= 0.80
    
    log_end(success, step_num, normalized_score, rewards_list)
    return normalized_score



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", default=USE_LOCAL_ENV)
    parser.add_argument("--no-llm", action="store_true", default=False)
    args = parser.parse_args()

    tasks_to_run = ["customer_issue_1", "customer_issue_2", "customer_issue_3"]
    client = _build_client() if not args.no_llm else None

    for task_id in tasks_to_run:
        try:
            run_episode(task_id, client, not args.no_llm, args.local)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")

if __name__ == "__main__":
    main()
