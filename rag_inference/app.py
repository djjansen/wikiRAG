"""FastAPI backend: serves the frontend and a /api/ask endpoint backed by rag.ask()."""
import csv
import hashlib
import hmac
import io
import logging
import os
from pathlib import Path
from urllib.parse import unquote

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wikirag")

STATIC_DIR = Path(__file__).parent / "static"

# Shared password for the login page. When unset, every login is refused rather than the app being left open.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
SESSION_COOKIE = "wikirag_session"
SESSION_MAX_AGE = 7 * 24 * 3600
# Paths reachable without logging in.
PUBLIC_PATHS = {"/login", "/health"}
indexed_articles = ["https://en.wikipedia.org/wiki/2018_Iraqi_parliamentary_election", 
                    "https://en.wikipedia.org/wiki/St._Louis_Blues",
                    "https://en.wikipedia.org/wiki/Saterland_Frisian_language",
                    "https://en.wikipedia.org/wiki/Peregrine_falcon",
                    "https://en.wikipedia.org/wiki/Transformer_(deep_learning)"
]

# Eval results written by rag_eval/run_eval.py: one CSV per version, with a version column and one column per metric.
EVAL_BUCKET = os.getenv("EVAL_BUCKET", "djj-wiki-rag")
EVAL_PREFIX = os.getenv("EVAL_PREFIX", "eval/")

if not APP_PASSWORD:
    log.warning("APP_PASSWORD is not set, so nobody can log in.")

app = FastAPI(title="wikiRAG")


def session_token() -> str:
    # Derived from the password, so changing APP_PASSWORD signs everyone out.
    return hmac.new(APP_PASSWORD.encode(), b"wikirag-session", hashlib.sha256).hexdigest()


def is_logged_in(request: Request) -> bool:
    cookie = request.cookies.get(SESSION_COOKIE, "")
    return bool(APP_PASSWORD) and hmac.compare_digest(cookie, session_token())


@app.middleware("http")
async def require_login(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS or is_logged_in(request):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Not logged in."}, status_code=401)
    return RedirectResponse("/login", status_code=303)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    number: int
    source: str
    label: str  # e.g. "St. Louis Blues > History"
    url: str  # Wikipedia section link, with a text fragment highlighting the quote when there is one
    quote: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]


class Article(BaseModel):
    title: str
    url: str


class VersionMetrics(BaseModel):
    version: str
    scores: dict[str, float]


class MetricsResponse(BaseModel):
    metrics: list[str]
    versions: list[VersionMetrics]


class LoginRequest(BaseModel):
    password: str = Field(max_length=200)


@app.get("/health")
def health():
    # Used by the ECS container health check.
    return {"status": "ok"}


@app.get("/login")
def login_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse("/", status_code=303)
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/login")
def login(req: LoginRequest, request: Request):
    if not APP_PASSWORD or not hmac.compare_digest(req.password.encode(), APP_PASSWORD.encode()):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    response = Response(status_code=204)
    response.set_cookie(
        SESSION_COOKIE,
        session_token(),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        # Cloudflare terminates TLS, so the scheme arrives in X-Forwarded-Proto rather than on the request itself.
        secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https",
    )
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/api/articles", response_model=list[Article])
def articles():
    # Title comes from the URL's last segment, e.g. .../wiki/St._Louis_Blues -> "St. Louis Blues".
    return [Article(title=unquote(url.rstrip("/").rsplit("/", 1)[-1]).replace("_", " "), url=url) for url in indexed_articles]


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


def version_key(version: str):
    # Sort "1.10" after "1.9"; anything non-numeric sorts after the numbered versions.
    try:
        return (0, tuple(int(part) for part in version.split(".")), version)
    except ValueError:
        return (1, (), version)


def load_eval_rows() -> list[dict[str, str]]:
    s3 = rag._s3_client()
    rows = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=EVAL_BUCKET, Prefix=EVAL_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                body = s3.get_object(Bucket=EVAL_BUCKET, Key=obj["Key"])["Body"].read().decode("utf-8-sig")
                rows.extend(csv.DictReader(io.StringIO(body)))
    return rows


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics():
    try:
        rows = load_eval_rows()
    except (ClientError, BotoCoreError):
        log.exception("Reading eval results from S3 failed")
        raise HTTPException(status_code=502, detail="Couldn't read the eval results. Check the server logs.")

    # Metric columns are everything but version and the unnamed pandas index, in the order they first appear.
    metric_names = []
    versions = {}
    for row in rows:
        version = (row.get("version") or "").strip()
        if not version:
            continue
        scores = versions.setdefault(version, {})
        for name, value in row.items():
            if not name or name == "version" or value in (None, ""):
                continue
            try:
                scores[name] = float(value)
            except ValueError:
                continue
            if name not in metric_names:
                metric_names.append(name)
    return MetricsResponse(
        metrics=metric_names,
        versions=[VersionMetrics(version=v, scores=versions[v]) for v in sorted(versions, key=version_key)],
    )


@app.get("/metrics")
def metrics_page():
    return FileResponse(STATIC_DIR / "metrics.html")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
