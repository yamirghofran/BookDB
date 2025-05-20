#!/usr/bin/env python3
"""
Script to upload embeddings to Qdrant vector database.
This script handles SBERT and GMF embeddings for books and users.
"""

import os
import sys
import argparse
import pandas as pd
import dask.dataframe as dd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def create_collection_if_not_exists(
    client, collection_name, vector_size, distance=Distance.COSINE
):
    """Create a collection in Qdrant if it doesn't already exist."""
    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print(f"Collection '{collection_name}' created.")


def combine_embeddings(row, num_cols=32):
    """Combine embedding columns into a single vector."""
    embedding_cols = [str(i) for i in range(num_cols)]
    return row[embedding_cols].tolist()


def upload_sbert_embeddings(client, embeddings_path):
    """Upload SBERT book embeddings to Qdrant."""
    if not os.path.exists(embeddings_path):
        print(f"SBERT embeddings file not found at {embeddings_path}")
        return False

    try:
        print("Processing SBERT book embeddings...")
        sbert_df = pd.read_parquet(embeddings_path)

        # SBERT embeddings are typically 768-dimensional
        create_collection_if_not_exists(client, "sbert_books", 768)

        # Prepare points for upsert
        points = []
        for _, row in sbert_df.iterrows():
            # Ensure book_id is a string
            book_id = str(row["book_id"])
            points.append(
                PointStruct(
                    id=book_id, vector=row["embedding"], payload={"book_id": book_id}
                )
            )

        # Batch upload points
        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name="sbert_books", points=batch, wait=True)

        print(f"Uploaded {len(points)} SBERT book embeddings.")
        return True
    except Exception as e:
        print(f"Error processing SBERT embeddings: {e}")
        return False


def upload_gmf_user_embeddings(client, embeddings_path, id_map_path=None):
    """Upload GMF user embeddings to Qdrant."""
    if not os.path.exists(embeddings_path):
        print(f"GMF user embeddings file not found at {embeddings_path}")
        return False

    try:
        print("Processing GMF user embeddings...")
        # Load GMF user embeddings
        gmf_user_df = dd.read_parquet(embeddings_path)
        gmf_user_df["embedding"] = gmf_user_df.apply(
            combine_embeddings, axis=1, meta=("embedding", "object")
        )
        gmf_user_df = gmf_user_df[["user_id", "embedding"]].compute()

        # Try to use ID map if available
        if id_map_path and os.path.exists(id_map_path):
            try:
                user_id_map = pd.read_csv(id_map_path)
                # Merge with ID map
                gmf_user_df = gmf_user_df.merge(
                    user_id_map, left_on="user_id", right_on="new_userId", how="inner"
                )
                gmf_user_df = gmf_user_df[["original_userId", "embedding"]].rename(
                    columns={"original_userId": "id"}
                )
                print("Successfully applied user ID mapping")
            except Exception as e:
                print(f"Failed to apply ID mapping: {e}, using original user IDs")
                gmf_user_df = gmf_user_df.rename(columns={"user_id": "id"})
        else:
            # No mapping file, use original IDs
            print("No user ID mapping file available, using original user IDs")
            gmf_user_df = gmf_user_df.rename(columns={"user_id": "id"})

        # GMF embeddings are typically 32-dimensional
        create_collection_if_not_exists(client, "gmf_users", 32)

        # Prepare points for upsert
        points = []
        for _, row in gmf_user_df.iterrows():
            user_id = str(row["id"])
            points.append(
                PointStruct(
                    id=user_id, vector=row["embedding"], payload={"user_id": user_id}
                )
            )

        # Batch upload points
        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name="gmf_users", points=batch, wait=True)

        print(f"Uploaded {len(points)} GMF user embeddings.")
        return True
    except Exception as e:
        print(f"Error processing GMF user embeddings: {e}")
        return False


def upload_gmf_book_embeddings(client, embeddings_path, id_map_path=None):
    """Upload GMF book embeddings to Qdrant."""
    if not os.path.exists(embeddings_path):
        print(f"GMF book embeddings file not found at {embeddings_path}")
        return False

    try:
        print("Processing GMF book embeddings...")
        # Load GMF book embeddings
        gmf_book_df = dd.read_parquet(embeddings_path)
        gmf_book_df["embedding"] = gmf_book_df.apply(
            combine_embeddings, axis=1, meta=("embedding", "object")
        )
        gmf_book_df = gmf_book_df[["item_id", "embedding"]].compute()

        # Try to use ID map if available
        if id_map_path and os.path.exists(id_map_path):
            try:
                item_id_map = pd.read_csv(id_map_path)
                # Merge with ID map
                gmf_book_df = gmf_book_df.merge(
                    item_id_map, left_on="item_id", right_on="new_itemId", how="inner"
                )
                gmf_book_df = gmf_book_df[["original_itemId", "embedding"]].rename(
                    columns={"original_itemId": "id"}
                )
                print("Successfully applied item ID mapping")
            except Exception as e:
                print(f"Failed to apply ID mapping: {e}, using original item IDs")
                gmf_book_df = gmf_book_df.rename(columns={"item_id": "id"})
        else:
            # No mapping file, use original IDs
            print("No item ID mapping file available, using original item IDs")
            gmf_book_df = gmf_book_df.rename(columns={"item_id": "id"})

        # GMF embeddings are typically 32-dimensional
        create_collection_if_not_exists(client, "gmf_books", 32)

        # Prepare points for upsert
        points = []
        for _, row in gmf_book_df.iterrows():
            book_id = str(row["id"])
            points.append(
                PointStruct(
                    id=book_id, vector=row["embedding"], payload={"book_id": book_id}
                )
            )

        # Batch upload points
        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name="gmf_books", points=batch, wait=True)

        print(f"Uploaded {len(points)} GMF book embeddings.")
        return True
    except Exception as e:
        print(f"Error processing GMF book embeddings: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload embeddings to Qdrant vector database"
    )
    parser.add_argument(
        "--qdrant-host",
        default=os.environ.get("QDRANT_HOST", "localhost"),
        help="Qdrant host (default: from QDRANT_HOST env var or 'localhost')",
    )
    parser.add_argument(
        "--qdrant-port",
        default=os.environ.get("QDRANT_PORT", "6333"),
        help="Qdrant port (default: from QDRANT_PORT env var or '6333')",
    )
    parser.add_argument(
        "--embeddings-dir",
        default="/tmp/embeddings",
        help="Directory containing embedding files (default: /tmp/embeddings)",
    )
    args = parser.parse_args()

    # Initialize Qdrant client
    client = QdrantClient(url=f"http://{args.qdrant_host}:{args.qdrant_port}")
    print(f"Connected to Qdrant at {args.qdrant_host}:{args.qdrant_port}")

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
    user_id_map_path = os.path.join(args.embeddings_dir, "user_id_map.csv")
    item_id_map_path = os.path.join(args.embeddings_dir, "item_id_map.csv")

    # Upload embeddings
    success_count = 0

    if upload_sbert_embeddings(client, sbert_embeddings_path):
        success_count += 1

    if upload_gmf_user_embeddings(client, gmf_user_embeddings_path, user_id_map_path):
        success_count += 1

    if upload_gmf_book_embeddings(client, gmf_book_embeddings_path, item_id_map_path):
        success_count += 1

    if success_count > 0:
        print(f"Successfully uploaded {success_count} embedding sets to Qdrant")
        return 0
    else:
        print("Failed to upload any embeddings to Qdrant")
        return 1


if __name__ == "__main__":
    sys.exit(main())
