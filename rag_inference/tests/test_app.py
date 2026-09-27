from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

import app as app_module
import rag

client = TestClient(app_module.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "wikiRAG" in res.text


def test_ask_returns_answer(monkeypatch):
    monkeypatch.setattr(rag, "ask", lambda q: {"answer": "Once, in 2019.", "sources": ["s3://b/k"]})
    res = client.post("/api/ask", json={"question": "  How many cups?  "})
    assert res.status_code == 200
    assert res.json() == {"question": "How many cups?", "answer": "Once, in 2019.", "sources": ["s3://b/k"]}


def test_ask_rejects_blank():
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_ask_bedrock_error_returns_502(monkeypatch):
    def boom(q):
        raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "RetrieveAndGenerate")

    monkeypatch.setattr(rag, "ask", boom)
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 502


def test_rag_parses_citations(monkeypatch):
    class FakeClient:
        def retrieve_and_generate(self, **kwargs):
            assert kwargs["input"] == {"text": "q"}
            return {
                "output": {"text": "answer"},
                "citations": [
                    {"retrievedReferences": [
                        {"location": {"s3Location": {"uri": "s3://b/a"}}},
                        {"location": {"s3Location": {"uri": "s3://b/a"}}},
                    ]},
                    {"retrievedReferences": [{"location": {"s3Location": {"uri": "s3://b/c"}}}]},
                ],
            }

    monkeypatch.setattr(rag, "RAG_MOCK", False)
    monkeypatch.setattr(rag, "_client", lambda: FakeClient())
    assert rag.ask("q") == {"answer": "answer", "sources": ["s3://b/a", "s3://b/c"]}
