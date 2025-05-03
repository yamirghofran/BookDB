#!/usr/bin/env python3
"""
download_r2_bookdb_users.py
---------------------------
Grabs the BookDB users PostgreSQL dump from Cloudflare R2.

Usage:
    python download_r2_bookdb_users.py           # saves to bookdb_users.sql
    R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… python download_r2_bookdb_users.py /tmp/users.sql
"""

import sys
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# --------- Configuration ---------------------------------------------------
ENDPOINT = "https://a9a190ee80813000e18bacf626b1281b.r2.cloudflarestorage.com"
BUCKET   = "bookdbio"
OBJECT_KEY = "db/bookdb_users.sql"

# Pull creds from env vars if set, else fall back to hard‑coded values
ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID",  "85fec6dd1268801ac8c1c59175ba0b76")
SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY",
                       "798b753bab748f2c7f5e0f46fd6506b7f0b206e362b1e00055d060a72b88d55d")

# Destination path (default: bookdb_users.sql in current dir)
DEST_PATH = sys.argv[1] if len(sys.argv) > 1 else "bookdb_users.sql"
# ---------------------------------------------------------------------------

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
