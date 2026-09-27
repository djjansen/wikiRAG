"""FastAPI backend: serves the frontend and a /api/ask endpoint backed by rag.ask()."""
import logging
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wikirag")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="wikiRAG")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    # Used by the ALB target group / ECS health check.
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Sync def: FastAPI runs it in a threadpool, so the blocking boto3 call won't stall the event loop.
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")
    try:
        result = rag.ask(question)
    except (ClientError, BotoCoreError):
        log.exception("Bedrock query failed")
        raise HTTPException(status_code=502, detail="The knowledge base query failed. Check the server logs.")
    return AskResponse(question=question, **result)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
