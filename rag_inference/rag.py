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
    """Run a retrieve-and-generate query. Returns {"answer": str, "sources": [str]}."""
    if RAG_MOCK:
        return {"answer": f"(mock answer) You asked: {question}", "sources": ["s3://mock-bucket/StLouisBlues.md-0"]}

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

    sources = []
    for citation in response.get("citations", []):
        for ref in citation.get("retrievedReferences", []):
            uri = ref.get("location", {}).get("s3Location", {}).get("uri")
            if uri and uri not in sources:
                sources.append(uri)

    return {"answer": response["output"]["text"], "sources": sources}


if __name__ == "__main__":
    print(ask("How many championships have the Blues won? When? Tell me the story")["answer"])
