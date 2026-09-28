"""FastAPI backend: serves the frontend and a /api/ask endpoint backed by rag.ask()."""
import hashlib
import hmac
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
indexed_articles = ["https://en.wikipedia.org/wiki/2018_Iraqi_parliamentary_election", "https://en.wikipedia.org/wiki/St._Louis_Blues"]

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
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]


class Article(BaseModel):
    title: str
    url: str


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


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
