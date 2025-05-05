import os
import math
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # For progress bar
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

BATCH_SIZE = 256  # Adjust based on your available RAM and description length
CHUNK_SIZE = 20000
OUTPUT_PATH = "book_texts_embeddings.parquet" # Directory to save parquet files
ID_COLUMN = "book_id" # The name of the column containing the book IDs
TEXT_COLUMN = "text" # The name of the column containing the text descriptions

def get_device():
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
    
device = get_device()


# Load model
model_path = '../sbert-output/finetuning-all-MiniLM-L6-v2-books'
print(f"Loading model from: {model_path}")
try:
    model = SentenceTransformer(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the model files are correctly placed in the 'model' directory.")

# Load Data
texts_df = dd.read_parquet("../data/book_texts.parquet")
texts_df = texts_df.compute()
texts_df.head()

# Generate embeddings in batches
# --- Validate Input DataFrame ---
if TEXT_COLUMN not in texts_df.columns:
    print(f"Error: Text column '{TEXT_COLUMN}' not found.")
    exit()
columns_to_keep = [col for col in texts_df.columns if col != TEXT_COLUMN]
if not columns_to_keep:
    print(f"Warning: No columns other than '{TEXT_COLUMN}' found to keep.")
    # Decide if this is an error or acceptable

# --- Process and Append in Chunks ---
print(f"\nProcessing {len(texts_df)} rows in chunks of {CHUNK_SIZE}...")
parquet_writer = None
total_rows_processed = 0

for i in range(0, len(texts_df), CHUNK_SIZE):
    chunk_df = texts_df.iloc[i:min(i + CHUNK_SIZE, len(texts_df))].copy() # Get a chunk
    print(f"  Processing chunk {i // CHUNK_SIZE + 1}/{math.ceil(len(texts_df) / CHUNK_SIZE)} (rows {i+1}-{min(i + CHUNK_SIZE, len(texts_df))})...")

    texts_in_chunk = chunk_df[TEXT_COLUMN].tolist()

    # Generate embeddings for the current chunk
    try:
        embeddings = model.encode(
            texts_in_chunk,
            batch_size=BATCH_SIZE,
            show_progress_bar=False, # Progress bar per chunk might be too verbose
            convert_to_numpy=True
        )
    except Exception as e:
        print(f"    Error encoding chunk: {e}")
        continue # Skip this chunk or handle error differently

    # Add embeddings to the chunk DataFrame
    if len(embeddings) == len(chunk_df):
        chunk_df['embedding'] = list(embeddings)
    else:
        print(f"    Error: Embedding count mismatch for chunk. Skipping write.")
        continue

    # Convert chunk DataFrame to Arrow Table
    try:
        # Explicitly define schema for embeddings if needed, especially for the first write
        # PyArrow usually infers it well, but being explicit can prevent issues.
        # Example schema definition (adjust dimensions):
        # fields = [pa.field(ID_COLUMN, pa.string()), pa.field(TEXT_COLUMN, pa.string()), pa.field('embedding', pa.list_(pa.float32()))]
        # schema = pa.schema(fields)
        # table = pa.Table.from_pandas(chunk_df, schema=schema, preserve_index=False)

        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
    except Exception as e:
        print(f"    Error converting chunk to Arrow Table: {e}")
        continue

    # Write/Append to Parquet file
    if parquet_writer is None:
        # Create writer on the first chunk, inferring schema from the first table
        try:
            parquet_writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
            print(f"    Created Parquet file: {OUTPUT_PATH} with schema:\n{table.schema}")
        except Exception as e:
            print(f"    Error creating Parquet file: {e}")
            exit() # Stop if file creation fails

    try:
        parquet_writer.write_table(table)
        total_rows_processed += len(chunk_df)
        print(f"    Appended {len(chunk_df)} rows to Parquet. Total written: {total_rows_processed}")
    except Exception as e:
        print(f"    Error writing chunk to Parquet: {e}")
        # Decide how to handle write errors (e.g., retry, log, stop)

# --- Close Parquet Writer ---
if parquet_writer:
    parquet_writer.close()
    print(f"\nFinished writing. Total rows processed: {total_rows_processed}")
else:
    print("\nNo data was written to the Parquet file.")

embeddings_df = dd.read_parquet(OUTPUT_PATH)
embeddings_df = embeddings_df.compute()
embeddings_df.head()


