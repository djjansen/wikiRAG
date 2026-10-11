import json
import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import utils.s3

BUCKET = 'djj-wiki-rag'

# small child chunks are what get embedded and retrieved
CHILD_CHUNK_SIZE = 800
CHILD_CHUNK_OVERLAP = 100

# a ## section is the parent returned for generation, unless it is longer than this,
# in which case each ### subsection becomes its own parent
PARENT_MAX_CHARS = 6000

# the "| --- | --- |" line that follows a markdown table's header row
TABLE_SEPARATOR = re.compile(r'^\|(\s*:?-+:?\s*\|)+$')


def is_header_only(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return all(line.lstrip().startswith('#') for line in lines)


def get_breadcrumb(title, headers):
    return ' > '.join(h for h in (title, headers.get('Header 2'), headers.get('Header 3')) if h)


def separate_tables(md_header_splits):
    # pull markdown tables out of each header split so they can be their own parents;
    # each table keeps the header metadata of the split it came from
    prose_docs = []
    tables = []
    for doc in md_header_splits:
        prose_lines = []
        table_lines = []
        # the trailing '' flushes a table that ends the split
        for line in doc.page_content.splitlines() + ['']:
            if line.strip().startswith('|'):
                table_lines.append(line.strip())
                continue
            if table_lines:
                # the header splitter drops blank lines, so adjacent tables arrive as one run of
                # | lines. Each table starts at the header row just above a separator line
                starts = [i - 1 for i in range(1, len(table_lines)) if TABLE_SEPARATOR.match(table_lines[i])]
                # | lines before the first header row aren't part of a table
                prose_lines.extend(table_lines[:starts[0]] if starts else table_lines)
                for start, end in zip(starts, starts[1:] + [len(table_lines)]):
                    tables.append(Document(page_content='\n'.join(table_lines[start:end]),
                                           metadata=dict(doc.metadata)))
                table_lines = []
            prose_lines.append(line)
        prose_docs.append(Document(page_content='\n'.join(prose_lines).strip(), metadata=dict(doc.metadata)))
    return prose_docs, tables


def group_into_parents(md_header_splits):
    # header splits come out in document order, so each section's splits are contiguous
    sections = []
    for doc in md_header_splits:
        key = (doc.metadata.get('Header 1'), doc.metadata.get('Header 2'))
        if sections and sections[-1]['key'] == key:
            sections[-1]['docs'].append(doc)
        else:
            sections.append({'key': key, 'docs': [doc]})

    parents = []
    for section in sections:
        section_name = section['key'][1]
        # drop splits that are just a header with no body (e.g. a ## followed directly by
        # another ##, or a header whose only content was a table)
        docs = [doc for doc in section['docs'] if not is_header_only(doc.page_content)]
        if not docs:
            continue

        section_len = sum(len(doc.page_content) for doc in docs)
        has_subsections = any('Header 3' in doc.metadata for doc in docs)
        if section_len <= PARENT_MAX_CHARS or not has_subsections:
            parents.append({
                'level': 'section' if section_name else 'intro',
                'section': section_name,
                'subsection': None,
                'docs': docs,
            })
            continue

        # section is too long: one parent per ### subsection. Any text between the ##
        # header and the first ### is attached to the first subsection so it isn't orphaned
        lead_docs = []
        current = None
        for doc in docs:
            subsection_name = doc.metadata.get('Header 3')
            if subsection_name is None and current is None:
                lead_docs.append(doc)
            elif current is None or current['subsection'] != subsection_name:
                current = {
                    'level': 'subsection',
                    'section': section_name,
                    'subsection': subsection_name,
                    'docs': lead_docs + [doc],
                }
                lead_docs = []
                parents.append(current)
            else:
                current['docs'].append(doc)

    return parents


def get_parent_text(parent):
    text = '\n\n'.join(doc.page_content for doc in parent['docs'])
    # a subsection parent should still say which ## section it belongs to
    if parent['level'] == 'subsection' and not text.startswith('## '):
        text = f"## {parent['section']}\n\n{text}"
    return text


def split_table_rows(table):
    # group rows into child-sized pieces, repeating the header row in every piece so
    # each one still reads as a table on its own
    lines = table.splitlines()
    header, rows = lines[:2], lines[2:]
    pieces = []
    current = []
    for row in rows:
        if current and len('\n'.join(header + current + [row])) > CHILD_CHUNK_SIZE:
            pieces.append('\n'.join(header + current))
            current = []
        current.append(row)
    pieces.append('\n'.join(header + current))
    return pieces


def build_parents(md_header_splits, title, text_splitter):
    # every parent carries the text returned for generation and the (piece, headers)
    # child chunks that point back at it
    prose_docs, tables = separate_tables(md_header_splits)

    parents = []
    for group in group_into_parents(prose_docs):
        pieces = [
            (piece, doc.metadata)
            for doc in group['docs']
            for piece in text_splitter.split_text(doc.page_content)
            if not is_header_only(piece)
        ]
        parents.append({'level': group['level'], 'text': get_parent_text(group), 'pieces': pieces})

    # a table is always returned whole, however many row chunks it was split into
    for table in tables:
        parents.append({
            'level': 'table',
            'text': f'{get_breadcrumb(title, table.metadata)}\n\n{table.page_content}',
            'pieces': [(piece, table.metadata) for piece in split_table_rows(table.page_content)],
        })

    return parents


def write_and_upload(text, name, folder):
    local_path = f'wiki_extract/files/{folder}/{name}'
    utils.s3.write_markdown_file(text, local_path)
    utils.s3.upload_to_s3(local_path, BUCKET, f'files/{folder}/{name}')


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

    # the extractor writes the page title as the # header
    base_filename = os.path.basename(file)
    title = next((doc.metadata['Header 1'] for doc in md_header_splits if 'Header 1' in doc.metadata),
                 base_filename.rsplit('.', 1)[0])

    # 3. Group the splits into parents (the text returned as context for generation) and
    # split each parent into small child chunks for retrieval. Tables are their own parents.
    # Using a recursive splitter ensures we fall back cleanly on paragraphs and sentences
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    parents = build_parents(md_header_splits, title, text_splitter)

    os.makedirs('wiki_extract/files/chunked', exist_ok=True)
    os.makedirs('wiki_extract/files/parents', exist_ok=True)

    # 4. Write parents to files/parents/ and children (plus a Bedrock metadata sidecar
    # pointing back at their parent) to files/chunked/
    child_index = 0
    for parent_index, parent in enumerate(parents):
        parent_name = f'{base_filename}-p{parent_index}'
        write_and_upload(parent['text'], parent_name, 'parents')

        for piece, headers in parent['pieces']:
            metadata = {
                'source_file': base_filename,
                'title': title,
                'parent_id': parent_name,
                'parent_key': f'files/parents/{parent_name}',
                'parent_level': parent['level'],
                'chunk_index': child_index,
            }
            if headers.get('Header 2'):
                metadata['section'] = headers['Header 2']
            if headers.get('Header 3'):
                metadata['subsection'] = headers['Header 3']

            # prefix each child with where it sits in the page, since a small chunk
            # often doesn't name its own subject
            chunk_name = f'{base_filename}-{child_index}'
            write_and_upload(f'{get_breadcrumb(title, headers)}\n\n{piece}', chunk_name, 'chunked')
            write_and_upload(json.dumps({'metadataAttributes': metadata}, ensure_ascii=False),
                             f'{chunk_name}.metadata.json', 'chunked')
            child_index += 1


chunk_md_file('wiki_extract/files/raw/SaterlandFrisianlanguage.md')
