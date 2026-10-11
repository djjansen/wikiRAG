import os

import boto3

s3 = boto3.client('s3')

BUCKET = 'djj-wiki-rag'

# Bump this to write a new, separate set of raw/chunked/parent files. Each version lives under
# its own top-level S3 prefix (e.g. v2/chunked/), so a knowledge base can point at one version
# without picking up the others or the original files/ set
DATA_VERSION = 'v2'


def local_path(folder, name):
    return f'wiki_extract/files/{DATA_VERSION}/{folder}/{name}'


def s3_key(folder, name):
    return f'{DATA_VERSION}/{folder}/{name}'


def upload_to_s3(filename, bucket_name, path):
    s3.upload_file(
    Filename=filename,     # Path to your local file
    Bucket=bucket_name,     # Name of your S3 bucket
    Key=path     # Desired path/filename inside S3
    )


def write_markdown_file(markdown_text, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(markdown_text)


def write_and_upload(text, folder, name):
    # write locally under the current data version, then upload to the matching S3 key
    path = local_path(folder, name)
    write_markdown_file(text, path)
    upload_to_s3(path, BUCKET, s3_key(folder, name))
