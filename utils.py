import yaml
import boto3
from botocore.config import Config
import os
from dotenv import load_dotenv
import torch
import logging

load_dotenv()

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        config = {} # Set config to empty dict or handle error appropriately
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = {} # Set config to empty dict or handle error appropriately
        return config
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        config = {}
        return config


def upload_to_r2(files, bucket_name):
    """
    Upload multiple files to Cloudflare R2 bucket
    
    Args:
        files (list): List of tuples containing (file_path, object_name)
        bucket_name (str): R2 bucket name
    """
    # Configure R2 client
    s3 = boto3.client('s3',
        endpoint_url = os.getenv("CLOUDFLARE_ENDPOINT_URL"),
        aws_access_key_id = os.getenv("CLOUDFLARE_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY"),
        config = Config(signature_version='s3v4')
    )

    for file_path, object_name in files:
        try:
            s3.upload_file(file_path, bucket_name, object_name)
            print(f"Successfully uploaded {file_path} to {object_name}")
        except Exception as e:
            print(f"Upload failed for {file_path}: {e}")

# Upload files
# files_to_upload = [
#     ("data/bookdb.sql", "db/bookdb.sql"),
# ]

#upload_to_r2(files_to_upload, "bookdbio")



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
        endpoint_url = os.getenv("CLOUDFLARE_ENDPOINT_URL"),
        aws_access_key_id = os.getenv("CLOUDFLARE_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY"),
        config = Config(signature_version='s3v4')
    )

    try:
        s3.download_file(bucket_name, object_name, local_path)
        print(f"Successfully downloaded {object_name} to {local_path}")
    except Exception as e:
        print(f"Download failed for {object_name}: {e}")

# Example usage:
# download_from_r2("data/reviews_dedup.parquet", "data/reviews_dedup.parquet")

def get_device() -> str:
    """Determines the best available device (MPS, CUDA, CPU)."""
    if torch.backends.mps.is_available():
        logging.info("Using Apple Metal Performance Shaders (MPS).")
        return "mps"
    elif torch.cuda.is_available():
        logging.info("Using CUDA.")
        return "cuda"
    else:
        logging.warning("Training on CPU will be very slow.")
        return "cpu"