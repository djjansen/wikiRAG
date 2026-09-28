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
        "redirects": True 
    }

    headers = {"User-Agent": "WikiRAGBot/1.0 "}
    response = requests.get(url=URL, params=PARAMS, headers=headers)
    html_content = response.text
    # Convert HTML string straight into structured Markdown
    markdown_text = md(html_content, heading_style="ATX", strip=["a", "img"])

    return markdown_text


title = "Saterland Frisian language"
md_text = collect_wiki_markdown(title)
clean_title = re.sub('[\W]+', '', title, count=0, flags=0)
filename = f"{clean_title}.md"
full_path = f"wiki_extract/files/raw/{filename}"
s3.write_markdown_file(md_text, full_path)
s3.upload_to_s3(full_path, 'djj-wiki-rag', f'files/raw/{filename}')