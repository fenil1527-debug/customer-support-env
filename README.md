---
title: Customer Support Agent Env
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: server/app.py
pinned: false
---

# Customer Support Reinforcement Environment

An **OpenEnv-compliant** agentic task where an AI customer support agent must juggle customer frustration while solving e-commerce tickets subject to strict financial budgets and company policies.

## Environment Description and Motivation

Customer support chatbots often hallucinate policies or blindly grant refunds. The purpose of this environment is to test an LLM's capacity to follow strict sequential tool-use boundaries while balancing mathematical constraints. 

Agents are evaluated on:
1. **Mathematical Constraints**: Agents receive a Daily Refund Budget (e.g., $500). Processing refunds dynamically drops this budget.
2. **Customer Patience**: The state tracks a `patience` metric. Duplicate messages or unhelpful queries dynamically lower the patience. If patience hits zero, the user abandons the chat.

## Action and Observation Space Definitions

### Observation Space
| Property | Type | Description |
|-----------|------|-------------|
| `user_query` | string | The active query from the simulated user. |
| `conversation_history` | list | List of prior message objects. |
| `step` | int | Current turn number (max 7). |
| `system_response` | string | JSON string containing responses from tools invoked. |
| `patience` | float | Customer patience level (starts at 1.0, decays mechanically). |
| `budget` | float | The agent's available daily financial budget. |

### Action Space
Agents must output strict JSON matching this schema:
| Field | Definition |
|-------|------------|
| `action_type` | Enforced enum: `"lookup_order"`, `"process_refund"`, `"lookup_kb"`, `"escalate_ticket"`, `"message_user"`. |
| `target` | The target ID (e.g. `order_id` or `topic`). |
| `response` | Text response to the customer. |

## Task Descriptions & Difficulty

The environment features 4 procedurally generated task families that heavily randomize variables to prevent memorization. 

| Task Family | Difficulty | Description |
|-------------|------------|-------------|
| `delivery` | Easy | Basic retrieval task where the agent looks up an order and returns the tracking via message. Tests basic multi-step logic. |
| `technical` | Easy | Non-order related query where the agent must query the Knowledge Base (KB) for specific system error codes (e.g., `error_504`). |
| `refund` | Medium | Agent must verify an order, verify the item eligibility, ensure the `cost` doesn't exceed the active `budget`, and execute a state-mutating `process_refund`. |
| `escalation` | Hard | A tricky scenario where an item is outside the standard return window. The agent must recognize a "Premium Policy" limit, refuse a direct refund, and `escalate_ticket` to a human manager. |

## Baseline Scores

The baseline heuristic agent (`inference.py --local --no-llm`) achieves the following scores against the automated Graders, producing scores from `0.0` to `1.0`:

- **Delivery**: 0.93
- **Refund**: 0.90
- **Technical**: 0.99
- **Escalation**: 0.99

*Note: The reward incorporates a scaling factor of `(patience ** 1.5)`, ensuring perfectly optimal execution paths are required to hit >0.90 scores.*

## Setup and Usage Instructions

**Prerequisites**: Docker or Python 3.10+

### Option 1: Docker (Hugging Face Spaces Native)
```bash
docker build -t customer-env .
docker run -p 7860:7860 customer-env
```

### Option 2: Local Python Execution
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Running the Baseline Agent
```bash
# Validating local environment heuristic clears constraints:
python inference.py --local --no-llm
```