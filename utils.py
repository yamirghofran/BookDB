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

# Example Usage (commented out):
# if __name__ == "__main__":
#     # Make sure to set your DISCORD_WEBHOOK_URL environment variable or replace the string directly
#     your_webhook_url = os.getenv("DISCORD_WEBHOOK_URL_TEST") 
#
#     if not your_webhook_url:
#         print("Please set the DISCORD_WEBHOOK_URL_TEST environment variable for testing.")
#     else:
#         # 1. Simple text message
#         # send_discord_webhook(your_webhook_url, content="Hello from Python script!")

#         # 2. Message with an embed
#         import datetime
#         example_embed = {
#             "title": "Pipeline Notification",
#             "description": "The data processing pipeline has **completed** successfully.",
#             "color": 0x00FF00,  # Green
#             "fields": [
#                 {"name": "Stage", "value": "Data Ingestion", "inline": True},
#                 {"name": "Status", "value": "Success ✅", "inline": True},
#                 {"name": "Items Processed", "value": "10,572"},
#                 {"name": "Next Step", "value": "Model Training"}
#             ],
#             "footer": {"text": f"Report generated at {datetime.datetime.now(datetime.timezone.utc).isoformat()}"},
#             "author": {"name": "Automated System", "icon_url": "https://i.imgur.com/R66g1Pe.jpg"} # Example icon
#         }
#         send_discord_webhook(your_webhook_url, embed=example_embed, username="My Bot", content="Update:")

#         # 3. Message with an embed and a file attachment
#         # Create a dummy file for testing
#         dummy_file_path = "test_report.txt"
#         with open(dummy_file_path, "w") as f:
#             f.write("This is a test report.\n")
#             f.write(f"Generated on: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
#
#         embed_with_file = {
#             "title": "Analysis Complete",
#             "description": "Analysis report is attached.",
#             "color": 0x0000FF # Blue
#         }
#         send_discord_webhook(
#             your_webhook_url,
#             content="Please find the report attached.",
#             embed=embed_with_file,
#             attachments=[('final_report.txt', dummy_file_path)]
#         )
#         os.remove(dummy_file_path) # Clean up dummy file

#         # 4. Message with only a file attachment
#         # Create another dummy file
#         dummy_image_path = "test_image.png" # Assuming you have a placeholder image or create one
#         # For simplicity, let's reuse the text file as a "pretend" image for this example structure
#         with open(dummy_image_path, "w") as f: f.write("This is a placeholder for an image.")
#
#         send_discord_webhook(
#             your_webhook_url,
#             attachments=[('important_diagram.png', dummy_image_path)],
#             content="Here is the diagram you requested:"
#         )
#         os.remove(dummy_image_path)


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