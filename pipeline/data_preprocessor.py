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

from .core import PipelineStep

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

    def _get_path(self, folder, file_name):
        return os.path.join(folder, file_name)

    def process_books(self) -> str:
        json_path = self._get_path(self.base_data_path, self.books_input_json)
        parquet_path = self._get_path(self.base_output_path, self.books_output_parquet)

        self.logger.info(f"Processing books from {json_path}...")
        books_df = pd.read_json(json_path, lines=True)
        books_df.to_parquet(parquet_path)
        self.logger.info(f"Books data saved to {parquet_path}")
        return parquet_path

    def process_interactions_csv(self) -> str:
        csv_path = self._get_path(self.base_data_path, self.interactions_csv_input)
        # book_id_map_path and user_id_map_path are now direct paths from config
        parquet_path = self._get_path(self.base_output_path, self.interactions_csv_output_parquet)

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
        return parquet_path

    def process_interactions_dedup_json(self) -> str:
        json_path = self._get_path(self.base_data_path, self.interactions_dedup_input_json)
        parquet_output_path = self._get_path(self.base_output_path, self.interactions_dedup_output_parquet)
        progress_tracker_path = self._get_path(self.base_output_path, self.interactions_dedup_progress_file)

        self.logger.info(f"Processing interactions dedup JSON from {json_path}...")
        start_chunk = 0
        if os.path.exists(progress_tracker_path):
            with open(progress_tracker_path) as f:
                content = f.read().strip()
                if content:
                    start_chunk = int(content)
        
        writer = None
        try:
            for chunk_count, chunk in enumerate(pd.read_json(json_path, lines=True, chunksize=self.interactions_dedup_chunk_size)):
                if chunk_count < start_chunk:
                    self.logger.info(f"Skipping chunk {chunk_count+1}/{start_chunk}...")
                    continue
                
                self.logger.info(f"Processing chunk {chunk_count + 1}...")
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    writer = pq.ParquetWriter(parquet_output_path, table.schema)
                
                writer.write_table(table)
                
                with open(progress_tracker_path, 'w') as f:
                    f.write(str(chunk_count + 1))
            self.logger.info(f"Interactions dedup JSON data saved to {parquet_output_path}")
        finally:
            if writer:
                writer.close()
                self.logger.info("Parquet writer closed for interactions_dedup.")
        
        return parquet_output_path

    def process_reviews(self) -> str:
        json_path = self._get_path(self.base_data_path, self.reviews_input_json)
        parquet_path = self._get_path(self.base_output_path, self.reviews_output_parquet)

        self.logger.info(f"Processing reviews from {json_path}...")
        reviews_df = pd.read_json(json_path, lines=True)
        reviews_df.to_parquet(parquet_path)
        self.logger.info(f"Reviews data saved to {parquet_path}")
        return parquet_path

    def process_book_works(self) -> str:
        json_path = self._get_path(self.base_data_path, self.book_works_input_json)
        parquet_path = self._get_path(self.base_output_path, self.book_works_output_parquet)

        self.logger.info(f"Processing book works from {json_path}...")
        books_works_df = pd.read_json(json_path, lines=True)
        books_works_df.to_parquet(parquet_path)
        self.logger.info(f"Book works data saved to {parquet_path}")
        return parquet_path

    def process_authors(self) -> str:
        json_path = self._get_path(self.base_data_path, self.authors_input_json)
        parquet_path = self._get_path(self.base_output_path, self.authors_output_parquet)

        self.logger.info(f"Processing authors from {json_path}...")
        authors_df = pd.read_json(json_path, lines=True)
        authors_df.to_parquet(parquet_path)
        self.logger.info(f"Authors data saved to {parquet_path}")
        return parquet_path

    def process(self) -> Dict[str, Any]:
        self.logger.info(f"Starting data preprocessing step: {self.name}")
        outputs = {}
        
        outputs["books_parquet_path"] = self.process_books()
        outputs["interactions_csv_parquet_path"] = self.process_interactions_csv()
        outputs["interactions_dedup_parquet_path"] = self.process_interactions_dedup_json()
        outputs["reviews_parquet_path"] = self.process_reviews()
        outputs["book_works_parquet_path"] = self.process_book_works()
        outputs["authors_parquet_path"] = self.process_authors()
        
        self.logger.info(f"Data preprocessing step {self.name} finished successfully.")
        self.output_data = outputs
        return outputs

    def run(self) -> Dict[str, Any]:
        """Satisfies PipelineStep ABC and delegates to process."""
        return self.process()