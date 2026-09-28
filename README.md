# wikiRAG
toy RAG-based question answering system for Wikipedia articles

## Web app (`rag_inference/`)

A FastAPI backend wraps the Bedrock Knowledge Base query in `rag.py` and serves a one-page frontend.

| Route | Purpose |
| --- | --- |
| `GET /` | Question and answer page |
| `GET /login`, `POST /login` | Password page. The POST takes `{"password": "..."}` and sets a session cookie |
| `POST /logout` | Clears the session cookie |
| `POST /api/ask` | Body `{"question": "..."}`, returns `{"question", "answer", "citations"}`. Each citation is `{"number", "source", "text"}`, and the answer carries matching `[n]` markers |
| `GET /health` | Health check for ECS |

Everything except `/login` and `/health` requires logging in with the shared password in `APP_PASSWORD`. If `APP_PASSWORD` is unset, nobody can log in.

Configuration comes from environment variables: `APP_PASSWORD`, `AWS_REGION`, `KNOWLEDGE_BASE_ID`, `MODEL_ARN`, `NUM_RESULTS`, and `RAG_MOCK` (set to `true` to return canned answers without calling AWS).

### Run locally

With Docker, using your local AWS credentials from `~/.aws`:

```
cd rag_inference
$env:APP_PASSWORD="<password>"
docker compose up --build
```

Open http://localhost:8080. Set `HOST_PORT` to use a different port, and `AWS_PROFILE` to pick a credentials profile.

Without Docker:

```
cd rag_inference
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
$env:APP_PASSWORD="<password>"
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
3. Store the login password as a SecureString parameter in the region the ECS service runs in, and add `rag_inference/ecs/execution-role-policy.json` as an inline policy on `ecsTaskExecutionRole` with `<ACCOUNT_ID>` replaced. ECS uses that policy to read the `/wikirag/*` parameters when the task starts.
   ```
   aws ssm put-parameter --name /wikirag/app-password --type SecureString --value <PASSWORD> --region us-east-2
   ```
4. Replace `<ACCOUNT_ID>` in `rag_inference/ecs/task-definition.json` and register it:
   ```
   aws ecs register-task-definition --cli-input-json file://rag_inference/ecs/task-definition.json --region us-east-2
   ```
5. Create an ECS service from the task definition with public IP turned on and a security group with no inbound rules. No load balancer is needed, because traffic arrives through the Cloudflare Tunnel described below.

### Cloudflare Tunnel

The task runs a `cloudflared` sidecar next to the app. It opens an outbound connection to Cloudflare, so the app is never exposed directly, and Cloudflare Access controls who can reach it.

1. In Cloudflare Zero Trust, go to **Networks > Tunnels**, create a Cloudflared tunnel, and copy its token. Add a public hostname whose service is `http://localhost:8000`. Containers in a Fargate task share localhost.
2. Store the token as a SecureString parameter, in the same region as the password. Use only the `eyJ...` token, not the whole install command the dashboard shows:
   ```
   aws ssm put-parameter --name /wikirag/cloudflared-token --type SecureString --value <TUNNEL_TOKEN> --region us-east-2
   ```
3. In Cloudflare Zero Trust, go to **Access > Applications** and add a self-hosted application for the same hostname, with a policy listing who may sign in.
