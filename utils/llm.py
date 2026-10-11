from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
import litellm
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.run_config import RunConfig
import os
from functools import lru_cache
import boto3
import instructor


# env vars for rag inference
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "7OP7AGPVXP")
MODEL_ARN = os.getenv("MODEL_ARN", "us.amazon.nova-pro-v1:0")
NUM_RESULTS = int(os.getenv("NUM_RESULTS", "5"))
# Set RAG_MOCK=true to return a canned answer without calling AWS (handy for frontend work).
RAG_MOCK = os.getenv("RAG_MOCK", "false").lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def _client():
    return boto3.client(service_name="bedrock-agent-runtime", region_name=AWS_REGION)

config = {
    "credentials_profile_name": "default",  # E.g "default"
    "region_name": "us-east-2",  # E.g. "us-east-1"
    "llm": "us.amazon.nova-pro-v1:0",  # E.g "anthropic.claude-3-5-sonnet-20241022-v2:0"
    "embeddings": "amazon.titan-embed-text-v2:0",  # E.g "amazon.titan-embed-text-v2:0"
    "temperature": 0.2,
}

generator_llm = LangchainLLMWrapper(ChatBedrockConverse(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    base_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
    model=config["llm"],
    temperature=config["temperature"],
))

generator_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    model_id=config["embeddings"],
))

# for eval
instructor_llm = llm_factory(
    "bedrock/us.anthropic.claude-sonnet-4-6",
    provider="litellm",
    client=litellm.acompletion,
    temperature=0.0,
    max_tokens=10000,
    top_p=None
)

# for eval
embeddings = embedding_factory(
    provider="litellm",
    model="bedrock/amazon.titan-embed-text-v2:0",
    client=litellm.aembedding
)

# 2. Force Ragas to run purely in a single-threaded sequential worker
sync_run_config = RunConfig(
    max_workers=1,  # Disables background async pooling
    timeout=180     # Keeps standard safety timeout per row
)


def bedrock_rag(question: str) -> dict:
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

    return response