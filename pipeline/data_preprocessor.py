import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from dotenv import load_dotenv
import pyarrow.parquet as pq
import pyarrow as pa
import dask.dataframe as dd
import boto3
from botocore.config import Config
from typing import Dict, Any
import datetime

from .core import PipelineStep
from utils import send_discord_webhook, download_initial_datasets

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class DataPreprocessorStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        self.base_data_path: str = "data/"
        self.base_output_path: str = "data/"
        
        # File name configurations - to be set in configure()
        self.books_input_json: str = "goodreads_books.json"
        self.books_output_parquet: str = "books.parquet"
        
        self.interactions_csv_input: str = "goodreads_interactions.csv"
        self.book_id_map_path: str = "data/book_id_map.csv" # Path relative to project root or absolute
        self.user_id_map_path: str = "data/user_id_map.csv" # Path relative to project root or absolute
        self.interactions_csv_output_parquet: str = "interactions.parquet"
        
        self.interactions_dedup_input_json: str = "goodreads_interactions_dedup.json"
        self.interactions_dedup_output_parquet: str = "interactions_dedup.parquet"
        self.interactions_dedup_progress_file: str = "chunk_progress.txt"
        self.interactions_dedup_chunk_size: int = 150000
        
        self.reviews_input_json: str = "goodreads_reviews.json"
        self.reviews_output_parquet: str = "reviews.parquet"
        
        self.book_works_input_json: str = "goodreads_book_works.json"
        self.book_works_output_parquet: str = "books_works.parquet"
        
        self.authors_input_json: str = "authors.json"
        self.authors_output_parquet: str = "authors.parquet"

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config) # Call base class configure if it does anything
        self.base_data_path = self.config.get("base_data_path", "data/")
        self.base_output_path = self.config.get("base_output_path", "data/processed_data/")
        
        # Ensure output directory exists
        os.makedirs(self.base_output_path, exist_ok=True)
        self.logger.info(f"Ensured output directory exists: {self.base_output_path}")

        # Check and download missing datasets
        self._check_and_download_datasets()

        # Specific file configurations
        self.books_input_json = self.config.get("books_input_json", self.books_input_json)
        self.books_output_parquet = self.config.get("books_output_parquet", self.books_output_parquet)
        
        self.interactions_csv_input = self.config.get("interactions_csv_input", self.interactions_csv_input)
        self.book_id_map_path = self.config.get("book_id_map_path", self.book_id_map_path)
        self.user_id_map_path = self.config.get("user_id_map_path", self.user_id_map_path)
        self.interactions_csv_output_parquet = self.config.get("interactions_csv_output_parquet", self.interactions_csv_output_parquet)
        
        self.interactions_dedup_input_json = self.config.get("interactions_dedup_input_json", self.interactions_dedup_input_json)
        self.interactions_dedup_output_parquet = self.config.get("interactions_dedup_output_parquet", self.interactions_dedup_output_parquet)
        self.interactions_dedup_progress_file = self.config.get("interactions_dedup_progress_file", self.interactions_dedup_progress_file)
        self.interactions_dedup_chunk_size = self.config.get("interactions_dedup_chunk_size", self.interactions_dedup_chunk_size)
        
        self.reviews_input_json = self.config.get("reviews_input_json", self.reviews_input_json)
        self.reviews_output_parquet = self.config.get("reviews_output_parquet", self.reviews_output_parquet)
        
        self.book_works_input_json = self.config.get("book_works_input_json", self.book_works_input_json)
        self.book_works_output_parquet = self.config.get("book_works_output_parquet", self.book_works_output_parquet)
        
        self.authors_input_json = self.config.get("authors_input_json", self.authors_input_json)
        self.authors_output_parquet = self.config.get("authors_output_parquet", self.authors_output_parquet)

    def _check_and_download_datasets(self):
        """Check for required datasets and download missing ones."""
        required_datasets = [
            "goodreads_books.json.gz", 
            "goodreads_book_works.json.gz", 
            "goodreads_reviews_dedup.json.gz", 
            "goodreads_interactions.csv", 
            "goodreads_interactions_dedup.json.gz",
            "book_id_map.csv", 
            "user_id_map.csv", 
            "goodreads_book_authors.json.gz"
        ]
        
        # Ensure data directory exists
        os.makedirs(self.base_data_path, exist_ok=True)
        
        missing_datasets = []
        for dataset in required_datasets:
            dataset_path = os.path.join(self.base_data_path, dataset)
            if not os.path.exists(dataset_path):
                missing_datasets.append(dataset)
                self.logger.info(f"Missing dataset: {dataset}")
        
        if missing_datasets:
            self.logger.info(f"Downloading {len(missing_datasets)} missing datasets...")
            
            # Send notification about missing datasets
            self._send_notification(
                "Downloading Missing Datasets",
                f"Found **{len(missing_datasets)}** missing datasets. Starting download...",
                color=0xFFA500,  # Orange for info
                fields=[
                    {"name": "Missing Files", "value": "\n".join([f"`{f}`" for f in missing_datasets[:5]]), "inline": False},
                    {"name": "Download Location", "value": f"`{self.base_data_path}`", "inline": True}
                ]
            )
            
            # Create a minimal file_names structure for the download function
            file_names_data = {
                'name': required_datasets,
                'type': ['complete'] * len(required_datasets)  # All datasets are in the complete category
            }
            
            try:
                download_initial_datasets(file_names_data, missing_datasets, self.base_data_path)
                
                # Send success notification
                self._send_notification(
                    "Dataset Download Complete",
                    f"Successfully downloaded **{len(missing_datasets)}** missing datasets",
                    color=0x00FF00,  # Green for success
                    fields=[
                        {"name": "Downloaded Files", "value": f"{len(missing_datasets)} files", "inline": True},
                        {"name": "Location", "value": f"`{self.base_data_path}`", "inline": True}
                    ]
                )
                
            except Exception as e:
                error_msg = f"Failed to download datasets: {str(e)}"
                self.logger.error(error_msg)
                self._send_notification(
                    "Dataset Download Failed",
                    error_msg,
                    error=True
                )
                raise
        else:
            self.logger.info("All required datasets are present.")

    def _get_path(self, folder, file_name):
        return os.path.join(folder, file_name)

    def _send_notification(self, title: str, description: str, color: int = 0x00FF00, fields: list = None, error: bool = False):
        """Send a Discord notification with consistent formatting."""
        try:
            embed = {
                "title": f"📊 {title}" if not error else f"❌ {title}",
                "description": description,
                "color": color if not error else 0xFF0000,  # Red for errors
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "footer": {"text": f"Pipeline Step: {self.name}"}
            }
            
            if fields:
                embed["fields"] = fields
                
            send_discord_webhook(
                content=None,
                embed=embed,
                username="BookDB Pipeline"
            )
        except Exception as e:
            self.logger.warning(f"Failed to send Discord notification: {e}")

    def process_books(self) -> str:
        json_path = self._get_path(self.base_data_path, self.books_input_json)
        parquet_path = self._get_path(self.base_output_path, self.books_output_parquet)

        try:
            self.logger.info(f"Processing books from {json_path}...")
            books_df = pd.read_json(json_path, lines=True)
            books_df.to_parquet(parquet_path)
            self.logger.info(f"Books data saved to {parquet_path}")
            
            # Send success notification
            self._send_notification(
                "Books Processing Complete",
                f"Successfully processed **{len(books_df):,}** books",
                fields=[
                    {"name": "Input File", "value": f"`{self.books_input_json}`", "inline": True},
                    {"name": "Output File", "value": f"`{self.books_output_parquet}`", "inline": True},
                    {"name": "Records", "value": f"{len(books_df):,}", "inline": True}
                ]
            )
            
            return parquet_path
        except Exception as e:
            error_msg = f"Failed to process books: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Books Processing Failed",
                error_msg,
                error=True
            )
            raise

    def process_interactions_csv(self) -> str:
        csv_path = self._get_path(self.base_data_path, self.interactions_csv_input)
        # book_id_map_path and user_id_map_path are now direct paths from config
        parquet_path = self._get_path(self.base_output_path, self.interactions_csv_output_parquet)

        try:
            self.logger.info(f"Processing interactions CSV from {csv_path}...")
            interactions_df = pd.read_csv(csv_path)

            # Use configured paths directly for map files
            self.logger.info(f"Loading book ID map from {self.book_id_map_path}")
            book_id_map_df = pd.read_csv(self.book_id_map_path)
            self.logger.info(f"Loading user ID map from {self.user_id_map_path}")
            user_id_map_df = pd.read_csv(self.user_id_map_path)

            interactions_df['user_id'] = interactions_df['user_id'].map(user_id_map_df.set_index('user_id_csv')['user_id'])
            interactions_df['book_id'] = interactions_df['book_id'].map(book_id_map_df.set_index('book_id_csv')['book_id'])
            
            interactions_df.to_parquet(parquet_path)
            self.logger.info(f"Interactions CSV data saved to {parquet_path}")
            
            # Send success notification
            self._send_notification(
                "Interactions CSV Processing Complete",
                f"Successfully processed **{len(interactions_df):,}** interactions with ID mapping",
                fields=[
                    {"name": "Input File", "value": f"`{self.interactions_csv_input}`", "inline": True},
                    {"name": "Output File", "value": f"`{self.interactions_csv_output_parquet}`", "inline": True},
                    {"name": "Records", "value": f"{len(interactions_df):,}", "inline": True}
                ]
            )
            
            return parquet_path
        except Exception as e:
            error_msg = f"Failed to process interactions CSV: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Interactions CSV Processing Failed",
                error_msg,
                error=True
            )
            raise

    def process_interactions_dedup_json(self) -> str:
        json_path = self._get_path(self.base_data_path, self.interactions_dedup_input_json)
        parquet_output_path = self._get_path(self.base_output_path, self.interactions_dedup_output_parquet)
        progress_tracker_path = self._get_path(self.base_output_path, self.interactions_dedup_progress_file)

        try:
            self.logger.info(f"Processing interactions dedup JSON from {json_path}...")
            start_chunk = 0
            if os.path.exists(progress_tracker_path):
                with open(progress_tracker_path) as f:
                    content = f.read().strip()
                    if content:
                        start_chunk = int(content)
            
            if start_chunk > 0:
                self._send_notification(
                    "Resuming Interactions Dedup Processing",
                    f"Resuming from chunk **{start_chunk + 1}**",
                    color=0xFFA500  # Orange for resume
                )
            
            writer = None
            total_records = 0
            try:
                for chunk_count, chunk in enumerate(pd.read_json(json_path, lines=True, chunksize=self.interactions_dedup_chunk_size)):
                    if chunk_count < start_chunk:
                        self.logger.info(f"Skipping chunk {chunk_count+1}/{start_chunk}...")
                        continue
                    
                    self.logger.info(f"Processing chunk {chunk_count + 1}...")
                    table = pa.Table.from_pandas(chunk)
                    total_records += len(chunk)
                    
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_output_path, table.schema)
                    
                    writer.write_table(table)
                    
                    with open(progress_tracker_path, 'w') as f:
                        f.write(str(chunk_count + 1))
                    
                    # Send progress update every 10 chunks
                    if (chunk_count + 1) % 10 == 0:
                        self._send_notification(
                            "Interactions Dedup Progress",
                            f"Processed **{chunk_count + 1}** chunks ({total_records:,} records so far)",
                            color=0x0099FF  # Blue for progress
                        )
                
                self.logger.info(f"Interactions dedup JSON data saved to {parquet_output_path}")
                
                # Send completion notification
                self._send_notification(
                    "Interactions Dedup Processing Complete",
                    f"Successfully processed **{total_records:,}** deduplicated interactions",
                    fields=[
                        {"name": "Input File", "value": f"`{self.interactions_dedup_input_json}`", "inline": True},
                        {"name": "Output File", "value": f"`{self.interactions_dedup_output_parquet}`", "inline": True},
                        {"name": "Total Records", "value": f"{total_records:,}", "inline": True},
                        {"name": "Chunk Size", "value": f"{self.interactions_dedup_chunk_size:,}", "inline": True}
                    ]
                )
                
            finally:
                if writer:
                    writer.close()
                    self.logger.info("Parquet writer closed for interactions_dedup.")
            
            return parquet_output_path
        except Exception as e:
            error_msg = f"Failed to process interactions dedup JSON: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Interactions Dedup Processing Failed",
                error_msg,
                error=True
            )
            raise

    def process_reviews(self) -> str:
        json_path = self._get_path(self.base_data_path, self.reviews_input_json)
        parquet_path = self._get_path(self.base_output_path, self.reviews_output_parquet)

        try:
            self.logger.info(f"Processing reviews from {json_path}...")
            reviews_df = pd.read_json(json_path, lines=True)
            reviews_df.to_parquet(parquet_path)
            self.logger.info(f"Reviews data saved to {parquet_path}")
            
            # Send success notification
            self._send_notification(
                "Reviews Processing Complete",
                f"Successfully processed **{len(reviews_df):,}** reviews",
                fields=[
                    {"name": "Input File", "value": f"`{self.reviews_input_json}`", "inline": True},
                    {"name": "Output File", "value": f"`{self.reviews_output_parquet}`", "inline": True},
                    {"name": "Records", "value": f"{len(reviews_df):,}", "inline": True}
                ]
            )
            
            return parquet_path
        except Exception as e:
            error_msg = f"Failed to process reviews: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Reviews Processing Failed",
                error_msg,
                error=True
            )
            raise

    def process_book_works(self) -> str:
        json_path = self._get_path(self.base_data_path, self.book_works_input_json)
        parquet_path = self._get_path(self.base_output_path, self.book_works_output_parquet)

        try:
            self.logger.info(f"Processing book works from {json_path}...")
            books_works_df = pd.read_json(json_path, lines=True)
            books_works_df.to_parquet(parquet_path)
            self.logger.info(f"Book works data saved to {parquet_path}")
            
            # Send success notification
            self._send_notification(
                "Book Works Processing Complete",
                f"Successfully processed **{len(books_works_df):,}** book works",
                fields=[
                    {"name": "Input File", "value": f"`{self.book_works_input_json}`", "inline": True},
                    {"name": "Output File", "value": f"`{self.book_works_output_parquet}`", "inline": True},
                    {"name": "Records", "value": f"{len(books_works_df):,}", "inline": True}
                ]
            )
            
            return parquet_path
        except Exception as e:
            error_msg = f"Failed to process book works: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Book Works Processing Failed",
                error_msg,
                error=True
            )
            raise

    def process_authors(self) -> str:
        json_path = self._get_path(self.base_data_path, self.authors_input_json)
        parquet_path = self._get_path(self.base_output_path, self.authors_output_parquet)

        try:
            self.logger.info(f"Processing authors from {json_path}...")
            authors_df = pd.read_json(json_path, lines=True)
            authors_df.to_parquet(parquet_path)
            self.logger.info(f"Authors data saved to {parquet_path}")
            
            # Send success notification
            self._send_notification(
                "Authors Processing Complete",
                f"Successfully processed **{len(authors_df):,}** authors",
                fields=[
                    {"name": "Input File", "value": f"`{self.authors_input_json}`", "inline": True},
                    {"name": "Output File", "value": f"`{self.authors_output_parquet}`", "inline": True},
                    {"name": "Records", "value": f"{len(authors_df):,}", "inline": True}
                ]
            )
            
            return parquet_path
        except Exception as e:
            error_msg = f"Failed to process authors: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Authors Processing Failed",
                error_msg,
                error=True
            )
            raise

    def process(self) -> Dict[str, Any]:
        self.logger.info(f"Starting data preprocessing step: {self.name}")
        
        # Send pipeline start notification
        self._send_notification(
            "Data Preprocessing Started",
            f"Beginning data preprocessing pipeline: **{self.name}**",
            color=0x0099FF,  # Blue for start
            fields=[
                {"name": "Input Directory", "value": f"`{self.base_data_path}`", "inline": True},
                {"name": "Output Directory", "value": f"`{self.base_output_path}`", "inline": True}
            ]
        )
        
        try:
            outputs = {}
            
            outputs["books_parquet_path"] = self.process_books()
            outputs["interactions_csv_parquet_path"] = self.process_interactions_csv()
            outputs["interactions_dedup_parquet_path"] = self.process_interactions_dedup_json()
            outputs["reviews_parquet_path"] = self.process_reviews()
            outputs["book_works_parquet_path"] = self.process_book_works()
            outputs["authors_parquet_path"] = self.process_authors()
            
            self.logger.info(f"Data preprocessing step {self.name} finished successfully.")
            
            # Send pipeline completion notification
            self._send_notification(
                "Data Preprocessing Complete! 🎉",
                f"All data processing tasks completed successfully for pipeline: **{self.name}**",
                color=0x00FF00,  # Green for success
                fields=[
                    {"name": "Books", "value": "✅ Complete", "inline": True},
                    {"name": "Interactions CSV", "value": "✅ Complete", "inline": True},
                    {"name": "Interactions Dedup", "value": "✅ Complete", "inline": True},
                    {"name": "Reviews", "value": "✅ Complete", "inline": True},
                    {"name": "Book Works", "value": "✅ Complete", "inline": True},
                    {"name": "Authors", "value": "✅ Complete", "inline": True},
                    {"name": "Output Directory", "value": f"`{self.base_output_path}`", "inline": False}
                ]
            )
            
            self.output_data = outputs
            return outputs
            
        except Exception as e:
            error_msg = f"Data preprocessing pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Data Preprocessing Pipeline Failed",
                error_msg,
                error=True
            )
            raise

    def run(self) -> Dict[str, Any]:
        """Satisfies PipelineStep ABC and delegates to process."""
        return self.process()