import io

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

import app as app_module
import rag

PASSWORD = "test-password"


@pytest.fixture(autouse=True)
def app_password(monkeypatch):
    monkeypatch.setattr(app_module, "APP_PASSWORD", PASSWORD)


@pytest.fixture
def anon():
    return TestClient(app_module.app)


@pytest.fixture
def client(anon):
    assert anon.post("/login", json={"password": PASSWORD}).status_code == 204
    return anon


def test_health_is_public(anon):
    assert anon.get("/health").json() == {"status": "ok"}


def test_login_page_is_public(anon):
    res = anon.get("/login")
    assert res.status_code == 200
    assert 'type="password"' in res.text


def test_index_redirects_to_login_when_logged_out(anon):
    res = anon.get("/", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/login"


def test_static_requires_login(anon):
    assert anon.get("/static/index.html", follow_redirects=False).status_code == 303


def test_api_returns_401_when_logged_out(anon):
    assert anon.post("/api/ask", json={"question": "hi"}).status_code == 401


def test_wrong_password_rejected(anon):
    assert anon.post("/login", json={"password": "nope"}).status_code == 401
    assert anon.get("/", follow_redirects=False).status_code == 303


def test_forged_cookie_rejected(anon):
    anon.cookies.set(app_module.SESSION_COOKIE, "forged")
    assert anon.get("/", follow_redirects=False).status_code == 303


def test_login_refused_when_password_unset(anon, monkeypatch):
    monkeypatch.setattr(app_module, "APP_PASSWORD", "")
    assert anon.post("/login", json={"password": ""}).status_code == 401
    assert anon.get("/", follow_redirects=False).status_code == 303


def test_index_served_when_logged_in(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "wikiRAG" in res.text


def test_logout_clears_session(client):
    res = client.post("/logout", follow_redirects=False)
    assert res.status_code == 303
    assert client.get("/", follow_redirects=False).status_code == 303


def test_articles_titles_from_urls(client, monkeypatch):
    monkeypatch.setattr(app_module, "indexed_articles", [
        "https://en.wikipedia.org/wiki/St._Louis_Blues",
        "https://en.wikipedia.org/wiki/Caf%C3%A9_society",
    ])
    assert client.get("/api/articles").json() == [
        {"title": "St. Louis Blues", "url": "https://en.wikipedia.org/wiki/St._Louis_Blues"},
        {"title": "Café society", "url": "https://en.wikipedia.org/wiki/Caf%C3%A9_society"},
    ]


def test_articles_requires_login(anon):
    assert anon.get("/api/articles").status_code == 401


CITATION = {
    "number": 1,
    "source": "s3://b/v2/parents/StLouisBlues.md-p3",
    "label": "St. Louis Blues > History",
    "url": "https://en.wikipedia.org/wiki/St._Louis_Blues#History:~:text=won%20in%202019",
    "quote": "won in 2019",
}


def test_ask_returns_answer(client, monkeypatch):
    citation = CITATION
    monkeypatch.setattr(rag, "ask", lambda q: {"answer": "Once, in 2019. [1]", "citations": [citation]})
    res = client.post("/api/ask", json={"question": "  How many cups?  "})
    assert res.status_code == 200
    assert res.json() == {"question": "How many cups?", "answer": "Once, in 2019. [1]", "citations": [citation]}


def test_ask_rejects_blank(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_ask_bedrock_error_returns_502(client, monkeypatch):
    def boom(q):
        raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "RetrieveAndGenerate")

    monkeypatch.setattr(rag, "ask", boom)
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 502


def test_rag_ask_returns_answer_and_citations_only(monkeypatch):
    monkeypatch.setattr(rag, "bedrock_rag",
                        lambda q: {"answer": "Once [1].", "citations": [CITATION], "contexts": ["whole section"]})
    assert rag.ask("q") == {"answer": "Once [1].", "citations": [CITATION]}


class FakeS3:
    """Serves list_objects_v2 / get_object from a {key: text} dict."""

    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in objects if k.startswith(Prefix)]}

        return Paginator()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key].encode())}


def test_metrics_page_requires_login(anon):
    assert anon.get("/metrics", follow_redirects=False).status_code == 303
    assert anon.get("/api/metrics").status_code == 401


def test_metrics_page_served_when_logged_in(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "Metrics" in res.text


def test_metrics_reads_eval_csvs(client, monkeypatch):
    monkeypatch.setattr(rag, "_s3_client", lambda: FakeS3({
        "eval/": "",
        "eval/version_1_10_eval.csv": ",version,faithfulness,relevancy,correctness\n0,1.10,0.9,0.8,0.7\n",
        "eval/version_1_0_eval.csv": ",version,faithfulness,relevancy,correctness\n0,1.0,0.81,0.87,0.65\n",
        "eval/version_1_9_eval.csv": ",version,faithfulness,relevancy,correctness\n0,1.9,0.5,,0.6\n",
        "eval/notes.txt": "ignored",
    }))
    assert client.get("/api/metrics").json() == {
        "metrics": ["faithfulness", "relevancy", "correctness"],
        "versions": [
            {"version": "1.0", "scores": {"faithfulness": 0.81, "relevancy": 0.87, "correctness": 0.65}},
            {"version": "1.9", "scores": {"faithfulness": 0.5, "correctness": 0.6}},
            {"version": "1.10", "scores": {"faithfulness": 0.9, "relevancy": 0.8, "correctness": 0.7}},
        ],
    }


def test_metrics_s3_error_returns_502(client, monkeypatch):
    class Broken:
        def get_paginator(self, name):
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "ListObjectsV2")

    monkeypatch.setattr(rag, "_s3_client", lambda: Broken())
    assert client.get("/api/metrics").status_code == 502
