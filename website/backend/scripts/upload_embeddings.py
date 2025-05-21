#!/usr/bin/env python3
"""
Script to upload embeddings to Qdrant vector database.
This script handles SBERT and GMF embeddings for books and users.
"""
import os
import csv
import sys
import uuid
import logging
import traceback
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

logging.basicConfig(level=logging.INFO)

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
        self.prefer_grpc = prefer_grpc

        if prefer_grpc:
            try:
                logging.info(
                    f"Connecting to Qdrant using gRPC at {self.host}:{self.grpc_port} (TLS disabled)"
                )
                self.client = QdrantClient(
                    host=self.host,
                    port=self.grpc_port,
                    prefer_grpc=True,
                    timeout=60.0,
                    https=False,
                )
                test_collection = "connection_test"
                try:
                    collections = self.client.get_collections().collections
                    collection_names = [c.name for c in collections]

                    if test_collection in collection_names:
                        self.client.delete_collection(collection_name=test_collection)

                    self.client.create_collection(
                        collection_name=test_collection,
                        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
                    )

                    test_uuid = str(uuid.uuid4())
                    test_point = PointStruct(
                        id=test_uuid,
                        vector=[1.0, 0.0, 0.0, 0.0],
                        payload={"test": True},
                    )
                    self.client.upsert(
                        collection_name=test_collection, points=[test_point]
                    )
                    retrieved = self.client.retrieve(
                        collection_name=test_collection, ids=[test_uuid]
                    )
                    if not retrieved:
                        raise Exception("Failed to retrieve test point")
                    self.client.delete_collection(collection_name=test_collection)
                    logging.info(
                        "Successfully verified gRPC connection with test operations"
                    )
                except Exception as e:
                    raise Exception(f"gRPC connection test failed: {e}")
            except Exception as e:
                logging.warning(
                    f"Failed to connect using gRPC: {e}, falling back to HTTP"
                )
                self.client = QdrantClient(
                    url=f"http://{self.host}:{self.port}",
                    timeout=60.0,
                    prefer_grpc=False,
                    https=False,
                )
        else:
            logging.info(
                f"Connecting to Qdrant using HTTP at {self.host}:{self.port} (TLS disabled)"
            )
            self.client = QdrantClient(
                url=f"http://{self.host}:{self.port}",
                timeout=60.0,
                prefer_grpc=False,
                https=False,
            )

        try:
            collections = self.client.get_collections()
            logging.info(
                f"Successfully connected to Qdrant. Available collections: {[c.name for c in collections.collections]}"
            )
            for collection in collections.collections:
                info = self.client.get_collection(collection.name)
                logging.info(
                    f"Collection {collection.name} contains {info.vectors_count} vectors"
                )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

    def create_collection_if_not_exists(
        self, collection_name, vector_size, distance_metric
    ):
        """Create a collection if it doesn't exist."""
        try:
            self.client.get_collection(collection_name)
            logging.info(f"Collection {collection_name} already exists")
        except Exception:
            logging.info(f"Creating collection {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance_metric),
            )
            logging.info(f"Created collection {collection_name}")

    def batch_upload_points(self, collection_name, points, batch_size=500):
        """Upload points in batches with proper error handling and verification."""
        if not points:
            logging.warning(f"No points to upload for collection '{collection_name}'.")
            return

        total_points = len(points)
        total_batches = (total_points + batch_size - 1) // batch_size
        logging.info(
            f"Uploading {total_points} points to {collection_name} in batches of {batch_size}"
        )

        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            batch_num = i // batch_size + 1
            try:
                self.client.upsert(
                    collection_name=collection_name, points=batch, wait=True
                )

                if self.prefer_grpc:
                    sample_point = batch[0]
                    retrieved = self.client.retrieve(
                        collection_name=collection_name, ids=[sample_point.id]
                    )
                    if not retrieved:
                        raise Exception(
                            f"Failed to verify upload of point {sample_point.id}"
                        )

                logging.info(
                    f"Successfully uploaded and verified batch {batch_num}/{total_batches}"
                )
            except Exception as e:
                logging.error(f"Error in batch {batch_num}: {e}")
                raise


class IdMapper:
    def __init__(self, user_id_map_path=None, item_id_map_path=None):
        self.user_id_map = {}
        self.item_id_map = {}
        if user_id_map_path and os.path.exists(user_id_map_path):
            self.user_id_map = self._load_id_map(user_id_map_path)
        if item_id_map_path and os.path.exists(item_id_map_path):
            self.item_id_map = self._load_id_map(item_id_map_path)

    def _load_id_map(self, path):
        id_map = {}
        try:
            with open(path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    internal_id, external_id = row
                    id_map[internal_id] = external_id
        except Exception as e:
            logging.error(f"Error loading ID map from {path}: {e}")
        return id_map

    def get_user_external_id(self, internal_id):
        return self.user_id_map.get(str(internal_id), str(internal_id))

    def get_item_external_id(self, internal_id):
        return self.item_id_map.get(str(internal_id), str(internal_id))


id_mapper = IdMapper()


class EmbeddingProcessor:
    def __init__(self, embedding_file_path, num_embedding_cols):
        self.embedding_file_path = embedding_file_path
        self.num_embedding_cols = num_embedding_cols
        self.raw_embeddings_df = None

    def process_embeddings(self, df):
        embedding_cols = [f"v{i}" for i in range(self.num_embedding_cols)]

        # Verify all embedding columns exist
        missing_cols = [col for col in embedding_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing embedding columns in DataFrame: {', '.join(missing_cols)}"
            )

        def combine_embeddings(row):
            return row[embedding_cols].tolist()

        df = df.copy()
        df["embedding"] = df.apply(combine_embeddings, axis=1)
        return df

    def load_and_process(self):
        # Load and validate raw data
        logging.info(f"Loading embeddings from {self.embedding_file_path}")
        self.raw_embeddings_df = pd.read_parquet(self.embedding_file_path)
        logging.info(f"Raw embeddings columns: {list(self.raw_embeddings_df.columns)}")

        # Process embeddings
        processed_df = self.process_embeddings(self.raw_embeddings_df)
        return processed_df


def upload_embeddings_to_qdrant(
    embeddings_df,
    collection_name,
    qdrant_client,
    batch_size=100,
):
    total_rows = len(embeddings_df)
    logging.info(f"Uploading {total_rows} embeddings to collection {collection_name}")

    # Process in batches
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = embeddings_df.iloc[start_idx:end_idx]

        points = []
        for _, row in batch_df.iterrows():
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}_{row['id']}")
            )
            point = PointStruct(
                id=point_id,
                vector=row["embedding"],
                payload={
                    "internal_id": str(row.get("internal_id", row["id"])),
                    "goodreads_id": str(row.get("goodreads_id", row["id"])),
                },
            )
            points.append(point)

        try:
            qdrant_client.upsert(
                collection_name=collection_name, wait=True, points=points
            )
            logging.info(f"Uploaded batch {start_idx + 1} to {end_idx} of {total_rows}")
        except Exception as e:
            logging.error(f"Error uploading batch to Qdrant: {e}")
            raise


def process_id_mappings(embeddings_df, user_id_map_path, item_id_map_path):
    """Process ID mappings for both users and items."""
    required_cols = ["internal_id", "goodreads_id"]
    merged_df = embeddings_df.copy()

    try:
        if user_id_map_path and os.path.exists(user_id_map_path):
            user_id_map_df = pd.read_csv(user_id_map_path)
            logging.info(f"User ID map shape: {user_id_map_df.shape}")

            missing_cols = [
                col for col in required_cols if col not in user_id_map_df.columns
            ]
            if missing_cols:
                logging.error(
                    f"Missing required columns in user ID map: {missing_cols}"
                )
                raise ValueError(f"Missing columns in user ID map: {missing_cols}")

            merged_df = pd.merge(
                merged_df,
                user_id_map_df,
                how="left",
                left_on="user_id",
                right_on="internal_id",
            )
    except Exception as e:
        logging.error(f"Error processing user ID mappings: {e}")
        raise

    try:
        if item_id_map_path and os.path.exists(item_id_map_path):
            item_id_map_df = pd.read_csv(item_id_map_path)
            logging.info(f"Item ID map shape: {item_id_map_df.shape}")

            missing_cols = [
                col for col in required_cols if col not in item_id_map_df.columns
            ]
            if missing_cols:
                logging.error(
                    f"Missing required columns in item ID map: {missing_cols}"
                )
                raise ValueError(f"Missing columns in item ID map: {missing_cols}")

            merged_df = pd.merge(
                merged_df,
                item_id_map_df,
                how="left",
                left_on="item_id",
                right_on="internal_id",
            )
    except Exception as e:
        logging.error(f"Error processing item ID mappings: {e}")
        raise

    return merged_df


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
        "--user-id-map",
        default="/data/id_maps/user_id_map.csv",
        help="Path to user ID mapping file",
    )
    parser.add_argument(
        "--item-id-map",
        default="/data/id_maps/item_id_map.csv",
        help="Path to item ID mapping file",
    )
    parser.add_argument(
        "--use-grpc",
        type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
        default="true",
        help="Use gRPC for Qdrant connection (true/false)",
    )

    try:
        args = parser.parse_args()

        # Initialize Qdrant client
        if args.use_grpc:
            client = QdrantClient(
                host=args.qdrant_host, port=args.qdrant_grpc_port, prefer_grpc=True
            )
        else:
            client = QdrantClient(
                host=args.qdrant_host, port=args.qdrant_port, prefer_grpc=False
            )

        # Create collections if they don't exist
        try:
            client.create_collection(
                collection_name="gmf_book_embeddings",
                vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            )
        except Exception as e:
            logging.warning(f"Collection might already exist: {e}")

        # Process and upload embeddings
        processor = EmbeddingProcessor(
            embedding_file_path=args.embeddings_dir, num_embedding_cols=4
        )

        embeddings_df = processor.load_and_process()
        if embeddings_df is not None:
            # Process ID mappings
            embeddings_df = process_id_mappings(
                embeddings_df, args.user_id_map, args.item_id_map
            )

            # Upload to Qdrant
            upload_embeddings_to_qdrant(
                embeddings_df=embeddings_df,
                collection_name="gmf_book_embeddings",
                qdrant_client=client,
            )
    except Exception as e:
        logging.error(f"Error during embedding upload process: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
