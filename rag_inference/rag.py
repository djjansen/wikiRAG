"""Query the Bedrock Knowledge Base and return a generated answer."""
import os
from functools import lru_cache

import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "7OP7AGPVXP")
MODEL_ARN = os.getenv("MODEL_ARN", "us.amazon.nova-pro-v1:0")
NUM_RESULTS = int(os.getenv("NUM_RESULTS", "5"))
# Set RAG_MOCK=true to return a canned answer without calling AWS (handy for frontend work).
RAG_MOCK = os.getenv("RAG_MOCK", "false").lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def _client():
    return boto3.client(service_name="bedrock-agent-runtime", region_name=AWS_REGION)


def ask(question: str) -> dict:
    """Run a retrieve-and-generate query.

    Returns {"answer": str, "citations": [{"number": int, "source": str, "text": str}]}. The answer
    carries [n] markers after each cited passage, where n is the citation's number.
    """
    if RAG_MOCK:
        return {
            "answer": f"(mock answer) You asked: {question} [1]",
            "citations": [{"number": 1, "source": "s3://mock-bucket/StLouisBlues.md-0", "text": "Mock source text."}],
        }

    response = _client().retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ARN,
                "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": NUM_RESULTS}},
            },
        },
    )

    answer = response["output"]["text"]
    citations = []
    numbers = {}  # (source, text) -> citation number, so a chunk cited twice keeps one number
    markers = []  # (position in answer, "[1][2]")
    cursor = 0
    for citation in response.get("citations", []):
        refs = []
        for ref in citation.get("retrievedReferences", []):
            source = ref.get("location", {}).get("s3Location", {}).get("uri", "")
            text = ref.get("content", {}).get("text", "")
            key = (source, text)
            if key not in numbers:
                numbers[key] = len(numbers) + 1
                citations.append({"number": numbers[key], "source": source, "text": text})
            if numbers[key] not in refs:
                refs.append(numbers[key])
        # Place the marker right after the cited passage. Matching its text is sturdier than
        # trusting the span offsets, whose end bound AWS does not document.
        part = citation.get("generatedResponsePart", {}).get("textResponsePart", {}).get("text", "")
        pos = answer.find(part, cursor) if part else -1
        if refs and pos != -1:
            cursor = pos + len(part)
            markers.append((cursor, "".join(f"[{n}]" for n in refs)))

    for pos, marker in reversed(markers):
        answer = f"{answer[:pos]} {marker}{answer[pos:]}"

    return {"answer": answer, "citations": citations}


if __name__ == "__main__":
    print(ask("How many championships have the Blues won? When? Tell me the story")["answer"])
