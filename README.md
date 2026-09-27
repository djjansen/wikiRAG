# wikiRAG
toy RAG-based question answering system for Wikipedia articles

## Web app (`rag_inference/`)

A FastAPI backend wraps the Bedrock Knowledge Base query in `rag.py` and serves a one-page frontend.

| Route | Purpose |
| --- | --- |
| `GET /` | Question and answer page |
| `POST /api/ask` | Body `{"question": "..."}`, returns `{"question", "answer", "sources"}` |
| `GET /health` | Health check for ECS and the load balancer |

Configuration comes from environment variables: `AWS_REGION`, `KNOWLEDGE_BASE_ID`, `MODEL_ARN`, `NUM_RESULTS`, and `RAG_MOCK` (set to `true` to return canned answers without calling AWS).

### Run locally

With Docker, using your local AWS credentials from `~/.aws`:

```
cd rag_inference
docker compose up --build
```

Open http://localhost:8080. Set `HOST_PORT` to use a different port, and `AWS_PROFILE` to pick a credentials profile.

Without Docker:

```
cd rag_inference
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\uvicorn app:app --reload --port 8080
```

Run the tests with `.venv\Scripts\python -m pytest`. They mock Bedrock, so they need no AWS access.

### Deploy to ECS (Fargate)

1. Create an ECR repository and push the image:
   ```
   aws ecr create-repository --repository-name wikirag --region us-east-2
   aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com
   docker build -t <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/wikirag:latest rag_inference
   docker push <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/wikirag:latest
   ```
2. Create an IAM task role named `wikirag-task-role` trusted by `ecs-tasks.amazonaws.com`, and attach `rag_inference/ecs/task-role-policy.json`. The app gets its AWS credentials from this role, so no keys go in the container.
3. Replace `<ACCOUNT_ID>` in `rag_inference/ecs/task-definition.json` and register it:
   ```
   aws ecs register-task-definition --cli-input-json file://rag_inference/ecs/task-definition.json --region us-east-2
   ```
4. Create an ECS service from the task definition behind an Application Load Balancer. Point the target group at container port 8000 with health check path `/health`. Retrieve-and-generate calls can take several seconds, so keep the ALB idle timeout at its default of 60 seconds or higher.
