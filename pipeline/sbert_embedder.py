import os
import math
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
# from tqdm import tqdm # Uncomment if you want to use tqdm progress bars
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

class SBertEmbedder:
    def __init__(self, input_texts_path, model_path, output_path,
                 batch_size=256, chunk_size=20000,
                 id_column="book_id", text_column="text"):
        self.input_texts_path = input_texts_path
        self.model_path = model_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.id_column = id_column
        self.text_column = text_column

        self.device = self._get_device()
        self.model = self._load_model()
        self.texts_df = None

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")


    def _get_device(self):
        """Checks for MPS (Apple Silicon GPU) availability, otherwise uses CPU."""
        if torch.backends.mps.is_available():
            print("MPS device found. Using Apple Silicon GPU.")
            return torch.device("mps")
        # elif torch.cuda.is_available(): # Uncomment if you might run on NVIDIA
        #     print("CUDA device found. Using NVIDIA GPU.")
        #     return torch.device("cuda")
        else:
            print("MPS (or CUDA) not available. Using CPU.")
            return torch.device("cpu")

    def _load_model(self):
        """Loads the SentenceTransformer model."""
        print(f"Loading model from: {self.model_path}")
        try:
            model = SentenceTransformer(self.model_path, device=self.device)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Ensure the model files are correctly placed in '{self.model_path}'.")
            raise  # Re-raise the exception to stop execution if model loading fails

    def _load_data(self):
        """Loads and preprocesses the input text data."""
        print(f"Loading data from: {self.input_texts_path}")
        try:
            self.texts_df = dd.read_parquet(self.input_texts_path).compute()
            print(f"Data loaded successfully. Shape: {self.texts_df.shape}")
            # print("Data head:")
            # print(self.texts_df.head())
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _validate_input_df(self):
        """Validates the loaded DataFrame."""
        if self.texts_df is None:
            print("Error: Data not loaded.")
            return False
        if self.text_column not in self.texts_df.columns:
            print(f"Error: Text column '{self.text_column}' not found in the DataFrame.")
            return False
        # columns_to_keep = [col for col in self.texts_df.columns if col != self.text_column]
        # if not columns_to_keep:
        #     print(f"Warning: No columns other than '{self.text_column}' found to keep.")
        return True

    def _process_and_save_embeddings(self):
        """Generates embeddings in chunks and saves them to a Parquet file."""
        if not self._validate_input_df():
            return

        print(f"\nProcessing {len(self.texts_df)} rows in chunks of {self.chunk_size}...")
        parquet_writer = None
        total_rows_processed = 0
        schema_defined = False

        for i in range(0, len(self.texts_df), self.chunk_size):
            chunk_df = self.texts_df.iloc[i:min(i + self.chunk_size, len(self.texts_df))].copy()
            print(f"  Processing chunk {i // self.chunk_size + 1}/{math.ceil(len(self.texts_df) / self.chunk_size)} (rows {i+1}-{min(i + self.chunk_size, len(self.texts_df))})...")

            texts_in_chunk = chunk_df[self.text_column].tolist()

            try:
                embeddings = self.model.encode(
                    texts_in_chunk,
                    batch_size=self.batch_size,
                    show_progress_bar=False, # Set to True for tqdm progress bar per chunk
                    convert_to_numpy=True,
                    device=self.device
                )
            except Exception as e:
                print(f"    Error encoding chunk: {e}")
                continue

            if len(embeddings) == len(chunk_df):
                chunk_df['embedding'] = list(embeddings)
            else:
                print(f"    Error: Embedding count mismatch for chunk ({len(embeddings)} vs {len(chunk_df)}). Skipping write.")
                continue
            
            # Drop the original text column after embeddings are generated to save space
            if self.text_column in chunk_df.columns:
                 chunk_df = chunk_df.drop(columns=[self.text_column])


            try:
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            except Exception as e:
                print(f"    Error converting chunk to Arrow Table: {e}")
                continue

            if parquet_writer is None:
                try:
                    parquet_writer = pq.ParquetWriter(self.output_path, table.schema)
                    schema_defined = True
                    print(f"    Created Parquet file: {self.output_path} with schema:\n{table.schema}")
                except Exception as e:
                    print(f"    Error creating Parquet file: {e}")
                    return # Stop if file creation fails

            if schema_defined:
                try:
                    parquet_writer.write_table(table)
                    total_rows_processed += len(chunk_df)
                    print(f"    Appended {len(chunk_df)} rows to Parquet. Total written: {total_rows_processed}")
                except Exception as e:
                    print(f"    Error writing chunk to Parquet: {e}")

        if parquet_writer:
            parquet_writer.close()
            print(f"\nFinished writing to {self.output_path}. Total rows processed: {total_rows_processed}")
        elif total_rows_processed == 0 :
            print("\nNo data was written to the Parquet file.")


    def run(self):
        """Executes the full embedding generation pipeline."""
        print("Starting SBERT embedding generation...")
        try:
            self._load_data()
            self._process_and_save_embeddings()
            print("SBERT embedding generation finished successfully.")
        except Exception as e:
            print(f"An error occurred during SBERT embedding generation: {e}")

if __name__ == '__main__':
    # Example usage:
    # Ensure the paths are correct for your environment.
    # Create dummy data and model for testing if needed.
    
    # Create a dummy input parquet file for testing
    if not os.path.exists("../data/dummy_book_texts.parquet"):
        os.makedirs("../data", exist_ok=True)
        dummy_data = {
            "book_id": [f"id_{i}" for i in range(50)],
            "text": [f"This is a sample description for book {i}." for i in range(50)],
            "other_column": [i * 10 for i in range(50)]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_parquet("../data/dummy_book_texts.parquet")
        print("Created dummy input file: ../data/dummy_book_texts.parquet")

    # Ensure a pre-trained model is available or use a standard one for testing
    # For this example, we'll assume 'all-MiniLM-L6-v2' is fine if custom path fails
    # The script expects a local path, so you might need to download it first
    # or provide a valid path to your fine-tuned model.
    test_model_path = '../sbert-output/finetuning-all-MiniLM-L6-v2-books' 
    # Fallback if the custom model path doesn't exist for local testing
    if not os.path.exists(test_model_path):
        print(f"Warning: Model path {test_model_path} not found. Using a default SBERT model for testing.")
        test_model_path = 'all-MiniLM-L6-v2' # A common model from sentence-transformers
        # This will download the model if not cached by sentence-transformers

    output_embeddings_path = "../data/processed_embeddings/dummy_book_texts_embeddings.parquet"

    embedder = SBertEmbedder(
        input_texts_path="../data/dummy_book_texts.parquet", # Use dummy data
        model_path=test_model_path, # Use your fine-tuned model path
        output_path=output_embeddings_path,
        batch_size=16, # Smaller for quick test
        chunk_size=25   # Smaller for quick test
    )
    embedder.run()

    # Verify output (optional)
    if os.path.exists(output_embeddings_path):
        print(f"\nVerifying output file: {output_embeddings_path}")
        try:
            embeddings_df = pd.read_parquet(output_embeddings_path)
            print("Output file head:")
            print(embeddings_df.head())
            print(f"Output file shape: {embeddings_df.shape}")
            if 'embedding' in embeddings_df.columns:
                print(f"Embedding dimension: {len(embeddings_df['embedding'].iloc[0])}")
            if 'text' not in embeddings_df.columns:
                print("Text column successfully removed from output.")
        except Exception as e:
            print(f"Error reading or verifying output parquet: {e}")
    else:
        print(f"Output file {output_embeddings_path} not found.")