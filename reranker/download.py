# Cell 1 - Download files with correct paths
import os
import boto3
from botocore.config import Config

# Get the absolute path of the BookDB directory

def download_from_r2(object_name, local_path, bucket_name="bookdbio"):
    """
    Download a file from Cloudflare R2 bucket
    
    Args:
        object_name (str): Object name in R2 bucket
        local_path (str): Local path to save file
        bucket_name (str): R2 bucket name
    """
    # Configure R2 client
    s3 = boto3.client('s3',
        endpoint_url = f"https://a9a190ee80813000e18bacf626b1281b.r2.cloudflarestorage.com/",
        aws_access_key_id = '85fec6dd1268801ac8c1c59175ba0b76',
        aws_secret_access_key = '798b753bab748f2c7f5e0f46fd6506b7f0b206e362b1e00055d060a72b88d55d',
        config = Config(signature_version='s3v4')
   )

    try:
        s3.download_file(bucket_name, object_name, local_path)
        print(f"Successfully downloaded {object_name} to {local_path}")
    except Exception as e:
        print(f"Download failed for {object_name}: {e}")

# Example usage:
# download_from_r2("data/reviews_dedup.parquet", "data/reviews_dedup.parquet")

# Download the required files with absolute paths
download_from_r2("data/books_dedup.parquet", "data/books_dedup.parquet")
