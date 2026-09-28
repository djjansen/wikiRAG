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
    processed_chunks = []

    curr_section = None
    curr_content = None
    for i, chunk in enumerate(chunk_texts):
        prefix = chunk[:3]
        if prefix == "## ":
            curr_section = i
            curr_content = chunk
        elif curr_content:
            if prefix == "###":
                if curr_section == i-1:
                    processed_chunks.append(curr_content+chunk)
                    curr_content = None
                    
        else:
            processed_chunks.append(chunk)


    for i, chunk in enumerate(processed_chunks):
        base_filename = file.split('/')[-1]
        chunk_name = f'wiki_extract/files/chunked/{base_filename}-{i}'
        utils.s3.write_markdown_file(chunk, chunk_name)
        utils.s3.upload_to_s3(chunk_name, 'djj-wiki-rag', f'files/chunked/{base_filename}-{i}')


chunk_md_file('wiki_extract/files/raw/2018Iraqiparliamentaryelection.md')