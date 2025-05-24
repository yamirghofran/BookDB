import yaml
import boto3
from botocore.config import Config
import os
from dotenv import load_dotenv
import torch
import logging
import requests
import json
from typing import Optional, List, Dict, Tuple

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


def send_discord_webhook(
    webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL"),
    content: Optional[str] = None,
    embed: Optional[Dict] = None,
    username: Optional[str] = "BookDB Pipeline",
    avatar_url: Optional[str] = None,
    attachments: Optional[List[Tuple[str, str]]] = None
):
    """
    Sends a message to a Discord channel via a webhook.

    Args:
        webhook_url (str): The Discord webhook URL.
        content (Optional[str]): The main text content of the message.
        embed (Optional[Dict]): A dictionary representing a Discord embed object.
                                 See Discord API docs for embed structure.
        username (Optional[str]): Override the default username of the webhook.
        avatar_url (Optional[str]): Override the default avatar of the webhook.
        attachments (Optional[List[Tuple[str, str]]]): A list of tuples,
            where each tuple is (filename_for_discord, local_file_path).
            Example: [('report.txt', '/path/to/local/report.txt')]
    """
    if not webhook_url:
        print("Error: Discord webhook URL is required.")
        return

    if not content and not embed and not attachments:
        print("No content, embed, or attachments to send. Message not sent.")
        return

    payload = {}
    if content:
        payload['content'] = content
    if username:
        payload['username'] = username
    if avatar_url:
        payload['avatar_url'] = avatar_url
    if embed:
        payload['embeds'] = [embed]  # Discord expects a list of embeds

    try:
        if attachments:
            files_to_send = {}
            opened_files = []
            try:
                # Prepare files for multipart upload
                for i, (discord_filename, local_filepath) in enumerate(attachments):
                    if not os.path.exists(local_filepath):
                        print(f"Warning: File not found at {local_filepath}, skipping attachment.")
                        continue
                    file_obj = open(local_filepath, 'rb')
                    opened_files.append(file_obj)
                    # The key here (e.g., 'file0', 'file1') is arbitrary but distinct for each file.
                    # Discord uses `discord_filename` for the actual attachment name.
                    files_to_send[f'file{i}'] = (discord_filename, file_obj)
                
                if not files_to_send and not payload.get('content') and not payload.get('embeds'):
                    print("No valid attachments to send, and no content or embed. Message not sent.")
                    return

                # Send payload_json along with files
                # The first element of the tuple for 'payload_json' can be None as filename is not relevant
                files_to_send['payload_json'] = (None, json.dumps(payload))
                response = requests.post(webhook_url, files=files_to_send)
            finally:
                for f_obj in opened_files:
                    f_obj.close()
        else:
            # Send as JSON if no attachments
            response = requests.post(webhook_url, json=payload)

        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        print(f"Message sent successfully to Discord. Status: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error sending message to Discord: {e}")
    except FileNotFoundError as e: # Should be caught above, but as a safeguard
        print(f"Error: Attachment file not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def download_initial_datasets(file_names_df, dataset_names: List[str] = ["goodreads_books.json.gz", "goodreads_book_works.json.gz", "goodreads_reviews_dedup.json.gz", "goodreads_interactions.csv", "goodreads_interactions_dedup.json.gz", "book_id_map.csv", "user_id_map.csv", "goodreads_book_authors.json.gz"], data_dir: str = "./data"):
    """
    Download initial datasets from remote URLs to local data directory.
    
    Args:
        file_names_df: DataFrame or dict-like object with 'name' and 'type' columns/keys
        dataset_names (List[str]): List of dataset names to download
        data_dir (str): Directory to save datasets (default: "./data")
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Create file name to type mapping
    if hasattr(file_names_df, 'values'):  # DataFrame-like
        file_name_type_mapping = dict(zip(file_names_df['name'].values, file_names_df['type'].values))
    else:  # Dictionary-like
        file_name_type_mapping = dict(zip(file_names_df['name'], file_names_df['type']))
    
    # Create file name to URL mapping
    file_name_url_mapping = {}
    for fname in file_name_type_mapping:
        ftype = file_name_type_mapping[fname]
        if ftype == "complete":
            url = 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/' + fname
            file_name_url_mapping[fname] = url
        elif ftype == "byGenre":
            url = 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/' + fname
            file_name_url_mapping[fname] = url
    
    def download_by_name(fname, local_filename):
        """Download a single dataset by name."""
        if fname in file_name_url_mapping:
            url = file_name_url_mapping[fname]
            try:
                print(f"Downloading {fname} from {url}...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f'Dataset {fname} has been downloaded to {local_filename}!')
            except requests.exceptions.RequestException as e:
                print(f'Error downloading {fname}: {e}')
            except Exception as e:
                print(f'Unexpected error downloading {fname}: {e}')
        else:
            print(f'Dataset {fname} cannot be found!')
    
    # Download each requested dataset
    for dataset_name in dataset_names:
        output_path = os.path.join(data_dir, dataset_name)
        download_by_name(dataset_name, output_path)


# Example Usage (commented out):
# if __name__ == "__main__":
#     # Make sure to set your DISCORD_WEBHOOK_URL environment variable or replace the string directly
#     your_webhook_url = os.getenv("DISCORD_WEBHOOK_URL_TEST")

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