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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

load_dotenv()

class DataProcessor:
    def __init__(self, base_data_path="data/", base_output_path="data/"):
        self.base_data_path = base_data_path
        self.base_output_path = base_output_path
        # Ensure output directory exists
        os.makedirs(self.base_output_path, exist_ok=True)

    def _get_path(self, folder, file_name):
        return os.path.join(folder, file_name)

    def process_books(self, input_json="goodreads_books.json", output_parquet="books_df.parquet"):
        json_path = self._get_path(self.base_data_path, input_json)
        parquet_path = self._get_path(self.base_output_path, output_parquet)

        print(f"Processing books from {json_path}...")
        books_df = pd.read_json(json_path, lines=True)
        books_df.to_parquet(parquet_path)
        print(f"Books data saved to {parquet_path}")
        return books_df

    def process_interactions_csv(self, 
                                 input_csv="goodreads_interactions.csv", 
                                 book_id_map_csv="book_id_map.csv", 
                                 user_id_map_csv="user_id_map.csv",
                                 output_parquet="interactions_df.parquet"):
        csv_path = self._get_path(self.base_data_path, input_csv)
        book_id_map_path = self._get_path(self.base_data_path, book_id_map_csv) # Assuming map files are also in base_data_path
        user_id_map_path = self._get_path(self.base_data_path, user_id_map_csv)
        parquet_path = self._get_path(self.base_output_path, output_parquet)

        print(f"Processing interactions CSV from {csv_path}...")
        interactions_df = pd.read_csv(csv_path)

        book_id_map_df = pd.read_csv(book_id_map_path)
        user_id_map_df = pd.read_csv(user_id_map_path)

        interactions_df['user_id'] = interactions_df['user_id'].map(user_id_map_df.set_index('user_id_csv')['user_id'])
        interactions_df['book_id'] = interactions_df['book_id'].map(book_id_map_df.set_index('book_id_csv')['book_id'])
        
        interactions_df.to_parquet(parquet_path)
        print(f"Interactions CSV data saved to {parquet_path}")
        return interactions_df

    def process_interactions_dedup_json(self, 
                                        input_json="goodreads_interactions_dedup.json", 
                                        output_parquet="interactions_dedup.parquet", 
                                        progress_file="chunk_progress.txt", 
                                        chunk_size=150000):
        json_path = self._get_path(self.base_data_path, input_json)
        parquet_output_path = self._get_path(self.base_output_path, output_parquet)
        progress_tracker_path = self._get_path(self.base_output_path, progress_file) # Progress file in output path

        print(f"Processing interactions dedup JSON from {json_path}...")
        start_chunk = 0
        if os.path.exists(progress_tracker_path):
            with open(progress_tracker_path) as f:
                content = f.read().strip()
                if content: # Ensure content is not empty
                    start_chunk = int(content)
        
        writer = None
        try:
            for chunk_count, chunk in enumerate(pd.read_json(json_path, lines=True, chunksize=chunk_size)):
                if chunk_count < start_chunk:
                    print(f"Skipping chunk {chunk_count+1}/{start_chunk}...")
                    continue
                
                print(f"Processing chunk {chunk_count + 1}...")
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    writer = pq.ParquetWriter(parquet_output_path, table.schema)
                
                writer.write_table(table)
                
                with open(progress_tracker_path, 'w') as f:
                    f.write(str(chunk_count + 1))
            print(f"Interactions dedup JSON data saved to {parquet_output_path}")
        finally:
            if writer:
                writer.close()
                print("Parquet writer closed.")
        
        # Optionally, return the path or a success status
        return parquet_output_path


    def process_reviews(self, input_json="goodreads_reviews.json", output_parquet="reviews_df.parquet"):
        json_path = self._get_path(self.base_data_path, input_json)
        parquet_path = self._get_path(self.base_output_path, output_parquet)

        print(f"Processing reviews from {json_path}...")
        reviews_df = pd.read_json(json_path, lines=True)
        reviews_df.to_parquet(parquet_path)
        print(f"Reviews data saved to {parquet_path}")
        return reviews_df

    def process_book_works(self, input_json="goodreads_book_works.json", output_parquet="books_works_df.parquet"):
        json_path = self._get_path(self.base_data_path, input_json)
        parquet_path = self._get_path(self.base_output_path, output_parquet)

        print(f"Processing book works from {json_path}...")
        books_works_df = pd.read_json(json_path, lines=True)
        books_works_df.to_parquet(parquet_path)
        print(f"Book works data saved to {parquet_path}")
        # books_works_df.head() # This would print to console, maybe return df and handle printing outside
        return books_works_df

    def process_authors(self, input_json="authors.json", output_parquet="authors_df.parquet"):
        json_path = self._get_path(self.base_data_path, input_json)
        parquet_path = self._get_path(self.base_output_path, output_parquet)

        print(f"Processing authors from {json_path}...")
        authors_df = pd.read_json(json_path, lines=True)
        authors_df.to_parquet(parquet_path)
        print(f"Authors data saved to {parquet_path}")
        return authors_df

    def run_pipeline(self):
        print("Starting data processing pipeline...")
        self.process_books()
        self.process_interactions_csv(book_id_map_csv="../data/book_id_map.csv", user_id_map_csv="../data/user_id_map.csv") # Adjusted path for map files
        self.process_interactions_dedup_json()
        self.process_reviews()
        self.process_book_works()
        self.process_authors()
        print("Data processing pipeline finished.")

if __name__ == "__main__":
    # Example usage:
    # Assumes your data files (goodreads_books.json, etc.) are in a 'data/' subdirectory 
    # relative to where you run the script, or you can specify absolute paths.
    # And map files are in '../data/' relative to the script.
    
    # If your script is in /Users/yamirghofran0/bookdbio/scripts/
    # and data is in /Users/yamirghofran0/bookdbio/data/
    # and map files are in /Users/yamirghofran0/bookdbio/data/ (adjust if they are truly in ../data relative to script)

    # Correct paths based on your project structure:
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # base_project_dir = os.path.dirname(script_dir) # This is /Users/yamirghofran0/bookdbio/
    # data_dir = os.path.join(base_project_dir, "data")
    # output_dir = os.path.join(base_project_dir, "data", "processed") # Example output subdir

    # processor = DataProcessor(base_data_path=data_dir, base_output_path=output_dir)
    
    # Simplified usage if script is run from /Users/yamirghofran0/bookdbio/
    # and data is in ./data/
    processor = DataProcessor(base_data_path="data/", base_output_path="data/processed_data/")
    # Make sure map files are correctly pathed in process_interactions_csv or adjust its parameters
    # For example, if map files are in 'data/' as well:
    # processor.process_interactions_csv(book_id_map_csv="book_id_map.csv", user_id_map_csv="user_id_map.csv")
    
    processor.run_pipeline()