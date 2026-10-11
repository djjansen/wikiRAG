"""Small-to-big RAG over the Bedrock Knowledge Base.

Only needs boto3, so it runs in the inference container; rag_eval imports bedrock_rag from here too.
"""
import os
import re
from functools import lru_cache
from urllib.parse import quote

import boto3

# env vars for rag inference
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "9UOJMBMQRX")
MODEL_ARN = os.getenv("MODEL_ARN", "us.amazon.nova-pro-v1:0")
# Small child chunks retrieved per question. Several usually share a parent, so this is
# higher than the number of parents we actually send to the model
NUM_RESULTS = int(os.getenv("NUM_RESULTS", "10"))
# Most parents (## sections, ### subsections or tables) given to the model as context
MAX_PARENTS = int(os.getenv("MAX_PARENTS", "5"))
# Set RAG_MOCK=true to return a canned answer without calling AWS (handy for frontend work).
RAG_MOCK = os.getenv("RAG_MOCK", "false").lower() in ("1", "true", "yes")

RAG_SYSTEM_PROMPT = """You answer questions about Wikipedia articles using only the sources provided.
Write the answer in your own words, as complete sentences.
After each claim, cite the source that supports it as its label followed by a short exact quote of the supporting words, copied verbatim from that source, in square brackets, e.g. [S1 "won the Stanley Cup in 2019"]. Cite several sources with several brackets, e.g. [S1 "..."][S3 "..."].
The quotes are only used to link to the source and are never shown to the reader, so the answer must make sense without them. For example:
The Blues won their first Stanley Cup in 2019. [S1 "won the Stanley Cup in 2019"]
Do not copy footnote numbers such as [12] from the sources into your answer.
If the sources do not contain the answer, say so rather than guessing."""
GENERATION_TEMPERATURE = 0.2
GENERATION_MAX_TOKENS = 2048

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/"
# a bracketed citation from the model: [S2 "quote"], [S2], or [S1, S3]
CITATION_MARKER = re.compile(r"(\s*)\[\s*(S\d+[^\]]*)\]")
CITATION_PART = re.compile(r"S(\d+)\s*:?\s*(?:[\"\u201c]([^\"\u201d]*)[\"\u201d])?")
# Wikipedia footnote markers like [12], which the model sometimes copies from the source text
FOOTNOTE_MARKER = re.compile(r"\[\d+\]")
# long quotes become a text fragment matching from their first to their last few words
FRAGMENT_EDGE_WORDS = 5


@lru_cache(maxsize=1)
def _client():
    return boto3.client(service_name="bedrock-agent-runtime", region_name=AWS_REGION)


@lru_cache(maxsize=1)
def _runtime_client():
    return boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)


@lru_cache(maxsize=1)
def _s3_client():
    return boto3.client(service_name="s3", region_name=AWS_REGION)


def _retrieve_parents(question: str) -> list[dict]:
    """Retrieve small child chunks and swap each for its parent: the whole ## section, ###
    subsection or table it came from. Parents are deduped and kept in the order of their
    best-ranked child."""
    response = _client().retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": NUM_RESULTS}},
    )

    parents = []
    seen = set()
    for result in response.get("retrievalResults", []):
        child_uri = result.get("location", {}).get("s3Location", {}).get("uri", "")
        metadata = result.get("metadata", {})
        parent_key = metadata.get("parent_key")
        # parents sit in the same bucket as the chunks that point at them
        bucket = child_uri.removeprefix("s3://").partition("/")[0]
        # a chunk with no parent (e.g. from a knowledge base over the original files/ set) is used as is
        source = f"s3://{bucket}/{parent_key}" if parent_key else child_uri
        if source in seen:
            continue
        seen.add(source)

        if parent_key:
            text = _s3_client().get_object(Bucket=bucket, Key=parent_key)["Body"].read().decode("utf-8")
        else:
            text = result.get("content", {}).get("text", "")
        # label and link the source by its article and the deepest heading the parent covers
        title = metadata.get("title")
        heading = metadata.get("section")
        if metadata.get("parent_level") in ("subsection", "table") and metadata.get("subsection"):
            heading = metadata["subsection"]
        label_parts = [title, metadata.get("section")]
        if heading != metadata.get("section"):
            label_parts.append(heading)
        label = " > ".join(part for part in label_parts if part)
        url = ""
        if title:
            url = WIKIPEDIA_URL + _wiki_anchor(title) + (f"#{_wiki_anchor(heading)}" if heading else "")

        parents.append({"source": source, "label": label, "url": url, "text": text})
        if len(parents) == MAX_PARENTS:
            break

    return parents


def _generate(question: str, parents: list[dict]) -> str:
    sources = "\n\n".join(
        f'<source label="S{i}" title="{parent["label"]}">\n{parent["text"]}\n</source>'
        for i, parent in enumerate(parents, start=1)
    )
    response = _runtime_client().converse(
        modelId=MODEL_ARN,
        system=[{"text": RAG_SYSTEM_PROMPT}],
        messages=[{
            "role": "user",
            "content": [{"text": f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"}],
        }],
        inferenceConfig={"temperature": GENERATION_TEMPERATURE, "maxTokens": GENERATION_MAX_TOKENS},
    )
    return "".join(block.get("text", "") for block in response["output"]["message"]["content"])


def _wiki_anchor(text: str) -> str:
    # Wikipedia page titles and section anchors use underscores for spaces
    return quote(text.replace("*", "").strip().replace(" ", "_"), safe="()_,'!.")


def _normalize(text: str) -> str:
    # compare quotes to sources ignoring footnote markers, markdown emphasis, whitespace and case
    return " ".join(FOOTNOTE_MARKER.sub("", text).replace("*", "").replace("`", "").split()).lower()


def _text_fragment(quoted: str) -> str:
    """Build a #:~:text= fragment that highlights the quote on the Wikipedia page."""
    # browsers match against the rendered page, so use the longest piece of the quote that
    # isn't broken up by footnote markers or table cell borders
    pieces = re.split(r"\[\d+\]|\|", quoted.replace("*", "").replace("`", ""))
    words = max(pieces, key=len).split()
    if not words:
        return ""

    def encode(part):
        # "-" and "," are text fragment syntax, so they must be escaped as well
        return quote(" ".join(part), safe="").replace("-", "%2D")

    if len(words) > 2 * FRAGMENT_EDGE_WORDS:
        return f"{encode(words[:FRAGMENT_EDGE_WORDS])},{encode(words[-FRAGMENT_EDGE_WORDS:])}"
    return encode(words)


def _citation_url(section_url: str, quoted: str) -> str:
    fragment = _text_fragment(quoted) if quoted and section_url else ""
    if not fragment:
        return section_url
    # a text fragment can follow a section anchor: the browser highlights the text if it
    # finds it and falls back to the section if not
    return f"{section_url}{':~:text=' if '#' in section_url else '#:~:text='}{fragment}"


def _number_citations(answer: str, parents: list[dict]) -> tuple[str, list[dict]]:
    """Replace the model's [S2 "quote"] citations with [n] markers numbered by first appearance.
    Each distinct (source, quote) pair gets its own number and a link to that span of the
    article, so the same section quoted twice yields two citations."""
    answer = FOOTNOTE_MARKER.sub("", answer)
    normalized_sources = [_normalize(parent["text"]) for parent in parents]
    numbers = {}  # (index into parents, quote) -> citation number
    citations = []

    def renumber(match, show_quotes=False):
        refs = []
        shown = []  # "quote [n]" pieces, used when the model answered only in quotes
        for label, quoted in CITATION_PART.findall(match.group(2)):
            index = int(label) - 1
            # skip labels for sources that don't exist
            if not 0 <= index < len(parents):
                continue
            quoted = " ".join(quoted.replace("*", "").split())
            written = quoted
            # only keep a quote that really is in the source; otherwise just link the section
            if quoted and _normalize(quoted) not in normalized_sources[index]:
                quoted = ""
            key = (index, quoted)
            if key not in numbers:
                numbers[key] = len(numbers) + 1
                parent = parents[index]
                citations.append({
                    "number": numbers[key],
                    "source": parent["source"],
                    "label": parent["label"],
                    "url": _citation_url(parent["url"], quoted),
                    "quote": quoted,
                })
            if numbers[key] not in refs:
                refs.append(numbers[key])
            if written:
                shown.append(f"{written} [{numbers[key]}]")
        if show_quotes and shown:
            return (match.group(1) or " ") + " ".join(shown)
        # keep the whitespace before the marker, unless every label in it was dropped
        return match.group(1) + "".join(f"[{n}]" for n in refs) if refs else ""

    cited = CITATION_MARKER.sub(renumber, answer)
    # The model sometimes answers only in quotes, which would leave nothing but [n] markers.
    # Show the quotes in that case (numbers are already assigned, so they come out the same)
    if not re.sub(r"\[\d+\]", "", cited).strip(" .\n"):
        cited = CITATION_MARKER.sub(lambda match: renumber(match, show_quotes=True), answer).strip()
    # adjacent citations that resolved to the same source and quote, e.g. [1][1]
    cited = re.sub(r"(\[\d+\])(?:\1)+", r"\1", cited)
    return cited, citations


def bedrock_rag(question: str) -> dict:
    """Small-to-big RAG: retrieve small chunks from the knowledge base, then generate from
    their parents (the full section, subsection or table each chunk came from).

    Returns {"answer": str, "citations": [{"number": int, "source": str, "label": str, "url": str,
    "quote": str}], "contexts": [str]}. The answer carries [n] markers after each cited passage,
    where n is the citation's number. Each citation's url links to the article section, with a
    text fragment highlighting the quoted span when the model's quote matched the source.
    contexts holds every parent text given to the model, for evaluation.
    """
    if RAG_MOCK:
        return {
            "answer": f"(mock answer) You asked: {question} [1]",
            "citations": [{
                "number": 1,
                "source": "s3://mock-bucket/v2/parents/StLouisBlues.md-p0",
                "label": "St. Louis Blues > History",
                "url": "https://en.wikipedia.org/wiki/St._Louis_Blues#History:~:text=Stanley%20Cup",
                "quote": "Stanley Cup",
            }],
            "contexts": ["Mock source text."],
        }

    parents = _retrieve_parents(question)
    if not parents:
        return {"answer": "I couldn't find anything about that in the indexed articles.", "citations": [], "contexts": []}

    answer, citations = _number_citations(_generate(question, parents), parents)
    return {"answer": answer, "citations": citations, "contexts": [parent["text"] for parent in parents]}

def ask(question: str) -> dict:
    """Answer a question for the web app.

    Returns {"answer": str, "citations": [{"number": int, "source": str, "label": str, "url": str,
    "quote": str}]}. The answer carries [n] markers after each cited passage, where n is the
    citation's number; each citation links to the cited span of the Wikipedia article.
    """
    result = bedrock_rag(question)
    return {"answer": result["answer"], "citations": result["citations"]}


if __name__ == "__main__":
    print(ask("How many championships have the Blues won? When? Tell me the story")["answer"])
