from bs4 import BeautifulSoup
import requests
from markdownify import markdownify as md
import re
import boto3
from utils import s3


def collect_wiki_markdown(wikipage_title):
    # Define parameters for plain text extraction
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "parse",
        "page": wikipage_title,
        "format": "json",
        "prop": "text",
        "redirects": True,
        "disableeditsection": True,  # drop the [edit] links next to each heading
        "formatversion": 2  # parse.text comes back as a plain HTML string
    }

    headers = {"User-Agent": "WikiRAGBot/1.0 "}
    response = requests.get(url=URL, params=PARAMS, headers=headers)
    response.raise_for_status()
    # Pull the page HTML out of the JSON response rather than converting the raw
    # response body, which leaves JSON escapes (\n, ä, \") in the text
    parsed = response.json()["parse"]
    html_content = parsed["text"]
    # Convert HTML string straight into structured Markdown
    # The text is read by the embedder and LLM rather than rendered, so don't backslash-escape
    # _ and * (which otherwise mangles LaTeX like d_{emb} and identifiers like Falco_peregrinus)
    markdown_text = md(html_content, heading_style="ATX", strip=["a", "img"],
                       escape_underscores=False, escape_asterisks=False)
    # Wikipedia uses non-breaking/thin spaces (e.g. "2,000\xa0(2015)"); the chunker's header
    # splitter drops non-printable characters, which would glue those words together
    markdown_text = re.sub(r"[  -   　]", " ", markdown_text)
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text).strip()

    # Page title as a top-level header so the chunker can use it as context
    return f"# {parsed['title']}\n\n{markdown_text}\n"


title = "Saterland Frisian language"
md_text = collect_wiki_markdown(title)
clean_title = re.sub('[\W]+', '', title, count=0, flags=0)
filename = f"{clean_title}.md"
full_path = f"wiki_extract/files/raw/{filename}"
s3.write_markdown_file(md_text, full_path)
s3.upload_to_s3(full_path, 'djj-wiki-rag', f'files/raw/{filename}')