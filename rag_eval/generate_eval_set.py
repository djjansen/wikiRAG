from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import DirectoryLoader

path = "wiki_extract/files/raw/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()


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

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=30)
# Convert to pandas first, then save with safe encoding
df = dataset.to_pandas()
df.to_json('rag_eval/files/test_set.jsonl', orient="records", lines=True, force_ascii=False)
