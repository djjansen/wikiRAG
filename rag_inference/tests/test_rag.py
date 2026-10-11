import io

import pytest

import rag

HISTORY = "## History\n\nThe Blues won the Stanley Cup[12] in 2019, their first championship.\n\nMore history."
TABLE = "St. Louis Blues > Franchise records > Scoring leaders\n\n| Player | Pts |\n| --- | --- |\n| Bernie Federko | 1,073 |"
PARENTS = {
    "v2/parents/StLouisBlues.md-p3": HISTORY,
    "v2/parents/StLouisBlues.md-p20": TABLE,
}


def _child(n, parent_key=None, level="section", section="History", subsection=None, text="chunk"):
    metadata = {"title": "St. Louis Blues", "section": section, "parent_level": level}
    if parent_key:
        metadata["parent_key"] = parent_key
    if subsection:
        metadata["subsection"] = subsection
    return {
        "content": {"text": text},
        "location": {"s3Location": {"uri": f"s3://djj-wiki-rag/v2/chunked/StLouisBlues.md-{n}"}},
        "metadata": metadata,
    }


class FakeAgent:
    def __init__(self, results):
        self.results = results

    def retrieve(self, **kwargs):
        assert kwargs["retrievalQuery"] == {"text": "q"}
        return {"retrievalResults": self.results}


class FakeS3:
    def __init__(self):
        self.reads = []

    def get_object(self, Bucket, Key):
        self.reads.append((Bucket, Key))
        return {"Body": io.BytesIO(PARENTS[Key].encode())}


class FakeRuntime:
    def __init__(self, answer):
        self.answer = answer
        self.prompt = None

    def converse(self, **kwargs):
        self.prompt = kwargs["messages"][0]["content"][0]["text"]
        return {"output": {"message": {"content": [{"text": self.answer}]}}}


@pytest.fixture
def fakes(monkeypatch):
    def install(results, answer):
        s3, runtime = FakeS3(), FakeRuntime(answer)
        monkeypatch.setattr(rag, "RAG_MOCK", False)
        monkeypatch.setattr(rag, "_client", lambda: FakeAgent(results))
        monkeypatch.setattr(rag, "_s3_client", lambda: s3)
        monkeypatch.setattr(rag, "_runtime_client", lambda: runtime)
        return s3, runtime
    return install


def test_children_are_swapped_for_deduped_parents(fakes):
    results = [
        _child(1, "v2/parents/StLouisBlues.md-p3"),
        _child(2, "v2/parents/StLouisBlues.md-p3"),
        _child(3, "v2/parents/StLouisBlues.md-p20", level="table", section="Franchise records", subsection="Scoring leaders"),
        _child(4, text="old chunk with no parent"),
    ]
    s3, runtime = fakes(results, "Answer [S1].")

    result = rag.bedrock_rag("q")

    # four children, three distinct parents, and only the two real parents are read from S3
    assert result["contexts"] == [HISTORY, TABLE, "old chunk with no parent"]
    assert s3.reads == [("djj-wiki-rag", "v2/parents/StLouisBlues.md-p3"), ("djj-wiki-rag", "v2/parents/StLouisBlues.md-p20")]
    assert 'label="S1" title="St. Louis Blues > History"' in runtime.prompt
    assert 'label="S2" title="St. Louis Blues > Franchise records > Scoring leaders"' in runtime.prompt


def test_stops_at_max_parents(fakes, monkeypatch):
    monkeypatch.setattr(rag, "MAX_PARENTS", 1)
    fakes([_child(1, "v2/parents/StLouisBlues.md-p3"), _child(2, "v2/parents/StLouisBlues.md-p20")], "x")
    assert rag.bedrock_rag("q")["contexts"] == [HISTORY]


def test_citations_link_to_quoted_spans(fakes):
    results = [
        _child(1, "v2/parents/StLouisBlues.md-p3"),
        _child(2, "v2/parents/StLouisBlues.md-p20", level="table", section="Franchise records", subsection="Scoring leaders"),
    ]
    answer = (
        'They won in 2019[12] [S1 "won the  *Stanley Cup* in 2019"]. '
        'Federko leads [S2 "not in the table"][S2]. '
        'Made up [S7 "x"]. '
        'Again [S1 "won the Stanley Cup in 2019"][S1 \u201cwon the Stanley Cup in 2019\u201d].'
    )
    fakes(results, answer)

    result = rag.bedrock_rag("q")

    # the footnote is dropped, [S7] doesn't exist, the unverifiable quote falls back to the
    # section link, and repeated identical citations collapse to one marker
    assert result["answer"] == "They won in 2019 [1]. Federko leads [2]. Made up. Again [1]."
    assert result["citations"] == [
        {
            "number": 1,
            "source": "s3://djj-wiki-rag/v2/parents/StLouisBlues.md-p3",
            "label": "St. Louis Blues > History",
            "url": "https://en.wikipedia.org/wiki/St._Louis_Blues#History:~:text=won%20the%20Stanley%20Cup%20in%202019",
            "quote": "won the Stanley Cup in 2019",
        },
        {
            "number": 2,
            "source": "s3://djj-wiki-rag/v2/parents/StLouisBlues.md-p20",
            "label": "St. Louis Blues > Franchise records > Scoring leaders",
            "url": "https://en.wikipedia.org/wiki/St._Louis_Blues#Scoring_leaders",
            "quote": "",
        },
    ]


def test_long_quote_uses_start_and_end_of_text_fragment():
    assert rag._text_fragment("one two three four five six seven eight nine ten eleven twelve") == (
        "one%20two%20three%20four%20five,eight%20nine%20ten%20eleven%20twelve"
    )
    # "-" and "," are text fragment syntax, and table borders split the quote
    assert rag._text_fragment("| 1,073 | well-known |") == "well%2Dknown"


def test_no_results_returns_message_without_calling_model(fakes):
    _, runtime = fakes([], "should not be used")
    result = rag.bedrock_rag("q")
    assert result["citations"] == [] and result["contexts"] == []
    assert runtime.prompt is None
