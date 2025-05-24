import os
import math
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
# from tqdm import tqdm # Uncomment if you want to use tqdm progress bars
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Any
from .core import PipelineStep
from ..utils import get_device

class SBertEmbedderStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_texts_path = "data/book_texts.parquet"
        self.model_path = "results/sbert"  # Updated to use results directory
        self.output_path = "embeddings/book_texts_embeddings.parquet"  # Updated to use embeddings directory
        self.batch_size = 256
        self.chunk_size = 20000
        self.id_column = "book_id"
        self.text_column = "text"
        self.device = get_device()
        self.model = None
        self.texts_df = None

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        # Update configuration from config dict
        for key, value in self.config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.device = get_device()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")

    def _load_model(self):
        """Load the fine-tuned SBERT model."""
        self.logger.info(f"Loading model from: {self.model_path}")
        try:
            if os.path.exists(self.model_path):
                self.model = SentenceTransformer(self.model_path, device=self.device)
                self.logger.info("Fine-tuned model loaded successfully.")
            else:
                self.logger.warning(f"Fine-tuned model not found at {self.model_path}, falling back to baseline model")
                # Fall back to baseline model if fine-tuned model not found
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self.logger.info("Baseline model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _load_data(self):
        """Load the book texts data."""
        self.logger.info(f"Loading data from: {self.input_texts_path}")
        try:
            if os.path.exists(self.input_texts_path):
                self.texts_df = pd.read_parquet(self.input_texts_path)
            else:
                # Try loading with dask and compute
                self.texts_df = dd.read_parquet(self.input_texts_path).compute()
            
            self.logger.info(f"Data loaded successfully. Shape: {self.texts_df.shape}")
            self.logger.info(f"Columns: {list(self.texts_df.columns)}")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _validate_input_df(self):
        """Validates the loaded DataFrame."""
        if self.texts_df is None:
            self.logger.error("Error: Data not loaded.")
            return False
        
        if self.text_column not in self.texts_df.columns:
            self.logger.error(f"Error: Text column '{self.text_column}' not found in the DataFrame.")
            self.logger.error(f"Available columns: {list(self.texts_df.columns)}")
            return False
        
        if self.id_column not in self.texts_df.columns:
            self.logger.warning(f"ID column '{self.id_column}' not found, will use index as ID.")
        
        return True

    def _process_and_save_embeddings(self):
        """Generates embeddings in chunks and saves them to a Parquet file."""
        if not self._validate_input_df():
            return 0

        self.logger.info(f"Processing {len(self.texts_df)} rows in chunks of {self.chunk_size}...")
        parquet_writer = None
        total_rows_processed = 0
        schema_defined = False

        for i in range(0, len(self.texts_df), self.chunk_size):
            chunk_df = self.texts_df.iloc[i:min(i + self.chunk_size, len(self.texts_df))].copy()
            self.logger.info(f"  Processing chunk {i // self.chunk_size + 1}/{math.ceil(len(self.texts_df) / self.chunk_size)} (rows {i+1}-{min(i + self.chunk_size, len(self.texts_df))})...")

            texts_in_chunk = chunk_df[self.text_column].tolist()

            try:
                embeddings = self.model.encode(
                    texts_in_chunk,
                    batch_size=self.batch_size,
                    show_progress_bar=True,  # Show progress for each chunk
                    convert_to_numpy=True,
                    device=self.device
                )
                self.logger.info(f"    Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            except Exception as e:
                self.logger.error(f"    Error encoding chunk: {e}")
                continue

            if len(embeddings) == len(chunk_df):
                chunk_df['embedding'] = list(embeddings)
            else:
                self.logger.error(f"    Error: Embedding count mismatch for chunk ({len(embeddings)} vs {len(chunk_df)}). Skipping write.")
                continue
            
            # Keep the ID column but drop the original text column to save space
            columns_to_keep = [col for col in chunk_df.columns if col != self.text_column]
            chunk_df = chunk_df[columns_to_keep]

            try:
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            except Exception as e:
                self.logger.error(f"    Error converting chunk to Arrow Table: {e}")
                continue

            if parquet_writer is None:
                try:
                    parquet_writer = pq.ParquetWriter(self.output_path, table.schema)
                    schema_defined = True
                    self.logger.info(f"    Created Parquet file: {self.output_path}")
                    self.logger.info(f"    Schema: {table.schema}")
                except Exception as e:
                    self.logger.error(f"    Error creating Parquet file: {e}")
                    return total_rows_processed

            if schema_defined:
                try:
                    parquet_writer.write_table(table)
                    total_rows_processed += len(chunk_df)
                    self.logger.info(f"    Appended {len(chunk_df)} rows to Parquet. Total written: {total_rows_processed}")
                except Exception as e:
                    self.logger.error(f"    Error writing chunk to Parquet: {e}")

        if parquet_writer:
            parquet_writer.close()
            self.logger.info(f"Finished writing to {self.output_path}. Total rows processed: {total_rows_processed}")
        elif total_rows_processed == 0:
            self.logger.warning("No data was written to the Parquet file.")
        
        return total_rows_processed

    def run(self) -> Dict[str, Any]:
        """Execute the full embedding generation pipeline."""
        self.logger.info("Starting SBERT embedding generation step...")
        outputs = {}
        
        try:
            # Load the fine-tuned model
            self._load_model()
            
            # Load the text data
            self._load_data()
            
            # Generate and save embeddings
            row_count = self._process_and_save_embeddings()
            
            outputs["embeddings_output_path"] = self.output_path
            outputs["row_count"] = row_count
            outputs["model_used"] = self.model_path
            outputs["embedding_dimension"] = self.model.get_sentence_embedding_dimension()
            
            self.logger.info(f"SBERT embedding generation finished successfully.")
            self.logger.info(f"Generated embeddings for {row_count} texts")
            self.logger.info(f"Embedding dimension: {outputs['embedding_dimension']}")
            self.logger.info(f"Output saved to: {self.output_path}")
            
        except Exception as e:
            self.logger.error(f"An error occurred during SBERT embedding generation: {e}")
            outputs["error"] = str(e)
        
        self.output_data = outputs
        return outputs