from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
import litellm
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.run_config import RunConfig
import instructor

# Models for building and scoring the eval set. The RAG pipeline itself lives in rag_inference/rag.py.

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
