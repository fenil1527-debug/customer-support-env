# Customer Support Env

OpenEnv-compatible customer support simulation for delivery, refund, technical, and escalation scenarios.

## Run

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Files

- `server/app.py`: FastAPI environment with `/reset`, `/step`, and `/state`
- `server/tasks.py`: Task generation and task families
- `server/graders.py`: Reward logic
- `inference.py`: Baseline agent runner

## Task Families

- `delivery`: missing shipment and tracking lookup
- `refund`: damaged item refund flow
- `technical`: app error triage and knowledge-base lookup
- `escalation`: premium return escalation flow