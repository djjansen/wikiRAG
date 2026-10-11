from utils import llm


"""Query the Bedrock Knowledge Base and return a generated answer."""
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

    response = llm.bedrock_rag(question)
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
