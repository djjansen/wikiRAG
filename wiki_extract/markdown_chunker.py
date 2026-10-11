from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import utils.s3


def chunk_md_file(file):
    # 1. Define the Markdown headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 2. Initialize the Markdown splitter
    # This splits the document by structural headings and retains them as metadata context
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )

    # Read your markdown file
    with open(file, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Perform the structural split
    md_header_splits = markdown_splitter.split_text(markdown_text)

    # 3. Sub-chunk large sections to fit model context limits (e.g., 500 tokens/characters)
    # Using a recursive splitter ensures we fall back cleanly on paragraphs and sentences
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=50
    )

    final_chunks = text_splitter.split_documents(md_header_splits)

    # 4. Extract plain text content and metadata
    chunk_texts = [chunk.page_content for chunk in final_chunks]
    # combine sub-sections immediately following sections
    # a "## " chunk is held until we see the next chunk: if that chunk is a "### "
    # subsection they are merged, otherwise the section is emitted on its own
    processed_chunks = []

    pending_section = None
    for chunk in chunk_texts:
        if pending_section is not None:
            if chunk.startswith("### "):
                processed_chunks.append(pending_section + "\n\n" + chunk)
                pending_section = None
                continue
            processed_chunks.append(pending_section)
            pending_section = None

        if chunk.startswith("## "):
            pending_section = chunk
        else:
            processed_chunks.append(chunk)

    # flush a trailing section that had nothing after it
    if pending_section is not None:
        processed_chunks.append(pending_section)


    for i, chunk in enumerate(processed_chunks):
        base_filename = file.split('/')[-1]
        chunk_name = f'wiki_extract/files/chunked/{base_filename}-{i}'
        utils.s3.write_markdown_file(chunk, chunk_name)
        utils.s3.upload_to_s3(chunk_name, 'djj-wiki-rag', f'files/chunked/{base_filename}-{i}')


chunk_md_file('wiki_extract/files/raw/SaterlandFrisianlanguage.md')