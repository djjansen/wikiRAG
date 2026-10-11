from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import DirectoryLoader
from utils import llm

path = "wiki_extract/files/raw/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()


generator = TestsetGenerator(llm=llm.generator_llm, embedding_model=llm.generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=30)
# Convert to pandas first, then save with safe encoding
df = dataset.to_pandas()
df.to_json('rag_eval/files/test_set.jsonl', orient="records", lines=True, force_ascii=False)
