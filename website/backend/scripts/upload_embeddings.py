#!/usr/bin/env python3
"""
Script to upload embeddings to Qdrant vector database.
This script handles SBERT and GMF embeddings for books and users.
"""
import os
import sys
import logging
import traceback
import pandas as pd
import dask.dataframe as dd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Docker environment configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
EMBEDDINGS_DIR = os.environ.get("EMBEDDINGS_DIR", "/tmp/embeddings")

# Ensure embeddings directory exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


class QdrantManager:
    def __init__(self, host=None, port=None, grpc_port=None, prefer_grpc=True):
        """Initialize QdrantManager with proper Docker networking support."""
        self.host = host or os.environ.get("QDRANT_HOST", "localhost")
        self.port = port or int(os.environ.get("QDRANT_PORT", "6333"))
        self.grpc_port = grpc_port or int(os.environ.get("QDRANT_GRPC_PORT", "6334"))

        if prefer_grpc:
            try:
                logging.info(
                    f"Connecting to Qdrant using gRPC at {self.host}:{self.grpc_port}"
                )
                self.client = QdrantClient(
                    host=self.host, port=self.grpc_port, prefer_grpc=True, timeout=60.0
                )
                logging.info("Successfully connected using gRPC")
            except Exception as e:
                logging.warning(
                    f"Failed to connect using gRPC: {e}, falling back to HTTP"
                )
                self.client = QdrantClient(
                    url=f"http://{self.host}:{self.port}", timeout=60.0
                )
        else:
            logging.info(f"Connecting to Qdrant using HTTP at {self.host}:{self.port}")
            self.client = QdrantClient(
                url=f"http://{self.host}:{self.port}", timeout=60.0
            )

        # Test connection
        try:
            collections = self.client.get_collections()
            logging.info(
                f"Successfully connected to Qdrant. Available collections: {collections.collections}"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

    def create_collection_if_not_exists(
        self, collection_name, vector_size, distance_metric
    ):
        try:
            self.client.get_collection(collection_name=collection_name)
            logging.info(f"Collection '{collection_name}' already exists.")
        except Exception:
            logging.info(f"Creating collection '{collection_name}'...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance_metric),
            )
            logging.info(f"Collection '{collection_name}' created.")

    def batch_upload_points(self, collection_name, points, batch_size=500):
        if not points:
            logging.warning(f"No points to upload for collection '{collection_name}'.")
            return False
        logging.info(
            f"Upserting {len(points)} points to '{collection_name}' in batches of {batch_size}..."
        )
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=collection_name, points=batch, wait=True
                )
            logging.info(
                f"Successfully uploaded {len(points)} points to '{collection_name}'."
            )
            return True
        except Exception as e:
            logging.error(f"Error uploading to {collection_name}: {e}")
            return False


class BaseEmbeddingProcessor(ABC):
    def __init__(self, embedding_file_path):
        self.embedding_file_path = embedding_file_path
        self.raw_embeddings_ddf = None
        self.processed_embeddings_ddf = None

    def _load_parquet_dask(self):
        logging.info(f"Loading embeddings from {self.embedding_file_path}...")
        self.raw_embeddings_ddf = dd.read_parquet(self.embedding_file_path)
        return self.raw_embeddings_ddf

    def _combine_embedding_columns(self, ddf, num_embedding_cols=32):
        embedding_cols = [str(i) for i in range(num_embedding_cols)]
        ddf["embedding"] = ddf.apply(
            lambda row: row[embedding_cols].tolist(),
            axis=1,
            meta=("embedding", "object"),
        )
        return ddf.drop(columns=embedding_cols)

    @abstractmethod
    def process_embeddings(self):
        pass

    @abstractmethod
    def generate_qdrant_points(self):
        pass


class GMFUserEmbeddingProcessor(BaseEmbeddingProcessor):
    def __init__(self, embedding_file_path, user_id_map_path, num_embedding_cols=32):
        super().__init__(embedding_file_path)
        self.user_id_map_path = user_id_map_path
        self.num_embedding_cols = num_embedding_cols

    def process_embeddings(self):
        self._load_parquet_dask()
        combined_ddf = self._combine_embedding_columns(
            self.raw_embeddings_ddf, self.num_embedding_cols
        )
        # If we have an ID map, use it, otherwise use user_id directly
        if self.user_id_map_path and os.path.exists(self.user_id_map_path):
            user_id_map_df = pd.read_csv(self.user_id_map_path)
            user_id_map_ddf = dd.from_pandas(user_id_map_df, npartitions=1)
            merged_ddf = combined_ddf.merge(
                user_id_map_ddf, left_on="user_id", right_on="new_userId", how="inner"
            )
            self.processed_embeddings_ddf = merged_ddf[
                ["original_userId", "embedding"]
            ].rename(columns={"original_userId": "id"})
        else:
            logging.info("No ID map provided or found, using user_id directly")
            self.processed_embeddings_ddf = combined_ddf[
                ["user_id", "embedding"]
            ].rename(columns={"user_id": "id"})
        logging.info("GMF User embeddings processed.")
        return self.processed_embeddings_ddf

    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError(
                "Embeddings not processed yet. Call process_embeddings() first."
            )
        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        for _, row in computed_df.iterrows():
            user_id_val = str(row["id"])
            points.append(
                PointStruct(
                    id=user_id_val,
                    vector=row["embedding"],
                    payload={"user_id": user_id_val},
                )
            )
        logging.info(f"Generated {len(points)} Qdrant points for GMF User embeddings.")
        return points


class GMFBookEmbeddingProcessor(BaseEmbeddingProcessor):
    def __init__(self, embedding_file_path, item_id_map_path, num_embedding_cols=32):
        super().__init__(embedding_file_path)
        self.item_id_map_path = item_id_map_path
        self.num_embedding_cols = num_embedding_cols

    def process_embeddings(self):
        self._load_parquet_dask()
        logging.info(
            f"Loaded {len(self.raw_embeddings_ddf.columns)} columns from GMF book embeddings"
        )
        logging.info(
            f"Available columns: {self.raw_embeddings_ddf.columns.compute().tolist()}"
        )

        combined_ddf = self._combine_embedding_columns(
            self.raw_embeddings_ddf, self.num_embedding_cols
        )
        logging.info(f"Combined embeddings shape: {combined_ddf.shape[0].compute()}")

        # If we have an ID map, use it, otherwise use item_id directly
        if self.item_id_map_path and os.path.exists(self.item_id_map_path):
            item_id_map_df = pd.read_csv(self.item_id_map_path)
            logging.info(f"Loaded ID map with {len(item_id_map_df)} entries")
            item_id_map_ddf = dd.from_pandas(item_id_map_df, npartitions=1)
            merged_ddf = combined_ddf.merge(
                item_id_map_ddf, left_on="item_id", right_on="new_itemId", how="inner"
            )
            logging.info(
                f"After merging with ID map: {merged_ddf.shape[0].compute()} entries"
            )
            self.processed_embeddings_ddf = merged_ddf[
                ["original_itemId", "embedding"]
            ].rename(columns={"original_itemId": "id"})
        else:
            logging.info("No ID map provided or found, using item_id directly")
            # Sample and log some IDs for debugging
            sample_ids = combined_ddf["item_id"].head().compute()
            logging.info(f"Sample of item_ids being used: {sample_ids.tolist()}")

            self.processed_embeddings_ddf = combined_ddf[
                ["item_id", "embedding"]
            ].rename(columns={"item_id": "id"})

        total_processed = self.processed_embeddings_ddf.shape[0].compute()
        logging.info(f"GMF Book embeddings processed. Total entries: {total_processed}")
        return self.processed_embeddings_ddf

    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError(
                "Embeddings not processed yet. Call process_embeddings() first."
            )
        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        correct_book_id_column_gmf = "id"

        # Log the first few IDs being processed
        sample_ids = computed_df[correct_book_id_column_gmf].head()
        logging.info(
            f"Sample of IDs being processed for Qdrant points: {sample_ids.tolist()}"
        )

        for _, row in computed_df.iterrows():
            try:
                point_id = str(
                    row[correct_book_id_column_gmf]
                )  # Keep as string to preserve UUID format
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=row["embedding"],
                        payload={"item_id": point_id},
                    )
                )
            except Exception as e:
                logging.error(
                    f"Error processing ID {row[correct_book_id_column_gmf]}: {e}"
                )
                continue

        logging.info(f"Generated {len(points)} Qdrant points for GMF Book embeddings")
        return points


class SBERTEmbeddingProcessor(BaseEmbeddingProcessor):
    def __init__(self, embedding_file_path):
        super().__init__(embedding_file_path)

    def process_embeddings(self):
        self._load_parquet_dask()
        if (
            "book_id" not in self.raw_embeddings_ddf.columns
            or "embedding" not in self.raw_embeddings_ddf.columns
        ):
            raise ValueError(
                "SBERT embeddings Dask DataFrame must contain 'book_id' and 'embedding' columns."
            )
        self.processed_embeddings_ddf = self.raw_embeddings_ddf[
            ["book_id", "embedding", "text"]
        ].rename(columns={"book_id": "id"})
        logging.info("SBERT Book embeddings processed.")
        return self.processed_embeddings_ddf

    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError(
                "Embeddings not processed yet. Call process_embeddings() first."
            )
        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        correct_book_id_column_sbert = "id"
        for _, row in computed_df.iterrows():
            try:
                point_id = int(row[correct_book_id_column_sbert])
            except ValueError:
                logging.warning(
                    f"Could not convert ID '{row[correct_book_id_column_sbert]}' to int for SBERT. Using as string."
                )
                point_id = str(row[correct_book_id_column_sbert])
            payload = {"book_id": point_id, "text": row.get("text", "")}
            points.append(
                PointStruct(id=point_id, vector=row["embedding"], payload=payload)
            )
        logging.info(
            f"Generated {len(points)} Qdrant points for SBERT Book embeddings."
        )
        return points


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload embeddings to Qdrant vector database"
    )
    parser.add_argument(
        "--qdrant-host",
        default=os.environ.get("QDRANT_HOST", "localhost"),
        help="Qdrant host address",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.environ.get("QDRANT_PORT", "6333")),
        help="Qdrant HTTP port",
    )
    parser.add_argument(
        "--qdrant-grpc-port",
        type=int,
        default=int(os.environ.get("QDRANT_GRPC_PORT", "6334")),
        help="Qdrant gRPC port",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=os.environ.get("EMBEDDINGS_DIR", "/tmp/embeddings"),
        help="Directory containing embedding files",
    )
    parser.add_argument(
        "--use-grpc",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Use gRPC for Qdrant connection (true/false)",
    )
    args = parser.parse_args()

    logging.info("Starting embedding upload with configuration:")
    logging.info(f"  Qdrant Host: {args.qdrant_host}")
    logging.info(f"  Qdrant HTTP Port: {args.qdrant_port}")
    logging.info(f"  Qdrant gRPC Port: {args.qdrant_grpc_port}")
    logging.info(f"  Embeddings Directory: {args.embeddings_dir}")
    logging.info(f"  Using gRPC: {args.use_grpc}")

    try:
        # Initialize connection to Qdrant
        qdrant_manager = QdrantManager(
            host=args.qdrant_host,
            port=args.qdrant_port,
            grpc_port=args.qdrant_grpc_port,
            prefer_grpc=args.use_grpc,
        )

        # Define embedding file paths
        sbert_embeddings_path = os.path.join(
            args.embeddings_dir, "SBERT_embeddings.parquet"
        )
        gmf_user_embeddings_path = os.path.join(
            args.embeddings_dir, "gmf_user_embeddings.parquet"
        )
        gmf_book_embeddings_path = os.path.join(
            args.embeddings_dir, "gmf_book_embeddings.parquet"
        )

        # Upload SBERT embeddings
        logging.info("Processing SBERT embeddings...")
        sbert_processor = SBERTEmbeddingProcessor(sbert_embeddings_path)
        sbert_processor.process_embeddings()
        sbert_points = sbert_processor.generate_qdrant_points()
        qdrant_manager.create_collection_if_not_exists(
            "sbert_books", vector_size=384, distance_metric=Distance.COSINE
        )
        qdrant_manager.batch_upload_points("sbert_books", sbert_points)

        # Upload GMF book embeddings
        logging.info("Processing GMF book embeddings...")
        gmf_book_processor = GMFBookEmbeddingProcessor(
            gmf_book_embeddings_path, None, num_embedding_cols=32
        )
        gmf_book_processor.process_embeddings()
        gmf_book_points = gmf_book_processor.generate_qdrant_points()
        qdrant_manager.create_collection_if_not_exists(
            "gmf_book_embeddings", vector_size=32, distance_metric=Distance.DOT
        )
        qdrant_manager.batch_upload_points("gmf_book_embeddings", gmf_book_points)

        # Upload GMF user embeddings
        logging.info("Processing GMF user embeddings...")
        gmf_user_processor = GMFUserEmbeddingProcessor(
            gmf_user_embeddings_path, None, num_embedding_cols=32
        )
        gmf_user_processor.process_embeddings()
        gmf_user_points = gmf_user_processor.generate_qdrant_points()
        qdrant_manager.create_collection_if_not_exists(
            "gmf_users", vector_size=32, distance_metric=Distance.DOT
        )
        qdrant_manager.batch_upload_points("gmf_users", gmf_user_points)

        # Verify the upload
        try:
            collections = qdrant_manager.client.get_collections()
            for collection in collections.collections:
                info = qdrant_manager.client.get_collection(collection.name)
                logging.info(
                    f"Collection {collection.name} contains {info.vectors_count} vectors"
                )
            logging.info("Embedding upload completed successfully")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Error verifying collections: {e}")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error during embedding upload: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
