#!/usr/bin/env python3
"""
download_r2_bookdb_users.py
---------------------------
Grabs the BookDB users PostgreSQL dump from Cloudflare R2.

Usage:
    python download_r2_bookdb_users.py           # saves to bookdb_users.sql
    python download_r2_bookdb_users.py /tmp/users.sql

Environment variables (should be defined in .env file):
    R2_ENDPOINT_URL - The Cloudflare R2 endpoint URL
    R2_BUCKET_NAME - The bucket name
    R2_OBJECT_KEY - Path to the object in the bucket
    R2_ACCESS_KEY_ID - Access key ID for authentication
    R2_SECRET_ACCESS_KEY - Secret access key for authentication
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Load environment variables from .env file
dotenv_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).joinpath('.env')
load_dotenv(dotenv_path)

# Get configuration from environment variables
ENDPOINT = os.getenv("R2_ENDPOINT_URL")
BUCKET = os.getenv("R2_BUCKET_NAME")
OBJECT_KEY = os.getenv("R2_OBJECT_KEY")
ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

# Validate required environment variables
required_vars = {
    "R2_ENDPOINT_URL": ENDPOINT,
    "R2_BUCKET_NAME": BUCKET,
    "R2_OBJECT_KEY": OBJECT_KEY,
    "R2_ACCESS_KEY_ID": ACCESS_KEY,
    "R2_SECRET_ACCESS_KEY": SECRET_KEY
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}", file=sys.stderr)
    print("Please create a .env file with the required variables", file=sys.stderr)
    sys.exit(1)

# Destination path (default: bookdb_users.sql in current dir)
DEST_PATH = sys.argv[1] if len(sys.argv) > 1 else "bookdb_users.sql"

def main() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

    try:
        print(f"Downloading s3://{BUCKET}/{OBJECT_KEY} → {DEST_PATH} …")
        s3.download_file(BUCKET, OBJECT_KEY, DEST_PATH)
    except ClientError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("✅  Finished!")

if __name__ == "__main__":
    main()
