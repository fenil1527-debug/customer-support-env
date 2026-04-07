# 🎧 Customer Support Simulation Environment

A lightweight **AI-powered customer support simulation environment** built with **FastAPI + OpenAI-compatible LLM inference**.

This project is designed to simulate realistic customer support conversations where an AI agent interacts with users facing issues like:

- 📦 Delayed deliveries
- 💸 Refund requests
- 🛠️ Technical app issues

The environment dynamically evaluates the AI’s responses based on **empathy, clarity, relevance, and actionability**, making it useful for **LLM behavior testing, reinforcement learning experiments, and prompt engineering practice**.

---

## 🚀 Features

- ✅ FastAPI-based simulation server
- ✅ Multiple realistic customer support tasks
- ✅ Dynamic multi-turn conversation flow
- ✅ Built-in reward scoring system
- ✅ Response grading using empathy + task relevance
- ✅ OpenAI / Hugging Face router compatible inference
- ✅ Multi-step agent validation loop
- ✅ Great for RL-style environment testing

---

## 🧠 How It Works

The project has **two major parts**:

### 1) Environment Server
The FastAPI server simulates customer support situations.

It provides endpoints like:

- `POST /reset` → starts a new customer issue
- `POST /step` → sends agent response and gets reward
- `GET /state` → returns current environment state

The environment randomly selects tasks such as:

- delivery delays
- refund requests
- technical crashes

It then scores the AI response based on:

- empathy words like *sorry* or *understand*
- useful actions like *refund*, *track*, *fix*
- task-specific keyword matching
- response clarity
- penalties for irrelevant replies

---

### 2) Inference Agent
The agent loop sends the user issue to an LLM and keeps interacting until:

- max steps are reached
- the reward becomes high enough
- the task is solved successfully

This simulates how an **AI support agent improves over multiple turns**.

---

## 🏗️ Project Structure

```bash
customer-support-env/
│
├── app.py              # FastAPI environment server
├── inference.py        # LLM agent interaction loop
├── requirements.txt
├── pyproject.toml
├── openenv.yaml
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd customer-support-env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Environment

Start the FastAPI server:

```bash
python app.py
```

Server runs on:

```bash
http://localhost:7860
```

---

## 🤖 Running the AI Agent

Set required environment variables:

```bash
export HF_TOKEN=your_token
export ENV_URL=http://localhost:7860
```

Then run:

```bash
python inference.py
```

---

## 📊 Reward Logic

The reward system scores every response between **0.0 → 1.0**.

### Scoring Factors
- 💙 Empathy
- 🎯 Task relevance
- 🔧 Problem-solving action
- 📝 Clear response length
- ⚠️ Penalties for weak answers
- 🚨 Priority-based boosts
- 🧩 Complexity-based boosts

This makes the environment useful for testing **real-world support quality**.

---

## 🌍 Example Use Cases

This project can be used for:

- LLM prompt engineering
- customer support bot evaluation
- RLHF-style reward experiments
- agent memory testing
- multi-turn conversation benchmarking
- fine-tuning support datasets

---

## 🔮 Future Improvements

Some great next upgrades:

- conversation sentiment scoring
- user frustration escalation
- ticket priority queues
- multilingual customer support
- CRM integration mock APIs
- advanced memory retention
- analytics dashboard

---

## 👨‍💻 Author

Built by **Fenil and Ridham** 🚀  
Focused on creating realistic **LLM training environments for customer support agents**.

---
