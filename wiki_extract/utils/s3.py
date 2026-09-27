import boto3

s3 = boto3.client('s3')


def upload_to_s3(filename, bucket_name, path):
    s3.upload_file(
    Filename=filename,     # Path to your local file
    Bucket=bucket_name,     # Name of your S3 bucket
    Key=path     # Desired path/filename inside S3
    )   


def write_markdown_file(markdown_text, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(markdown_text)