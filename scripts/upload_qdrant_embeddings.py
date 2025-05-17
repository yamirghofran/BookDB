# Uploading Embeddings to Qdrand
# In this notebook, we will upload the following embeddings to qdrant
# - SBERT Book Metadata embeddings
# - GMF User embeddings
# - GMF Book embeddings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import dask.dataframe as dd

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# 1. Remap GMF Embeddings to original User and Item ids
user_id_map = pd.read_csv('../data/user_id_map_reduced.csv')
item_id_map = pd.read_csv('../data/item_id_map_reduced.csv')

gmf_user_embeddings_df = dd.read_parquet("../embeddings/gmf_user_embeddings.parquet")

# Identify the embedding columns (assuming they are '0' to '31')
embedding_cols = [str(i) for i in range(32)]

# Define a function to combine embedding columns into a list
def combine_embeddings(row):
    return row[embedding_cols].tolist()

# Apply the function row-wise to create the 'embedding' column
# meta specifies the output column name and data type for Dask
gmf_user_embeddings_df['embedding'] = gmf_user_embeddings_df.apply(
    combine_embeddings,
    axis=1,
    meta=('embedding', 'object')
)

# Select the user_id and the new embedding column, dropping the old ones
gmf_user_embeddings_final = gmf_user_embeddings_df[['user_id', 'embedding']]


# Merge user embeddings with user ID map
gmf_user_embeddings_final = gmf_user_embeddings_final.merge(
    user_id_map,
    left_on='user_id',
    right_on='new_userId',
    how='inner'
)

# Select and rename columns
gmf_user_embeddings_final = gmf_user_embeddings_final[['original_userId', 'embedding']]
gmf_user_embeddings_final = gmf_user_embeddings_final.rename(columns={'original_userId': 'user_id'})


gmf_book_embeddings_df = dd.read_parquet("../embeddings/gmf_book_embeddings.parquet")
gmf_book_embeddings_df.head()


# Identify the embedding columns (assuming they are '0' to '31')
embedding_cols = [str(i) for i in range(32)]

# Define a function to combine embedding columns into a list
def combine_embeddings(row):
    return row[embedding_cols].tolist()

# Apply the function row-wise to create the 'embedding' column
# meta specifies the output column name and data type for Dask
gmf_book_embeddings_df['embedding'] = gmf_book_embeddings_df.apply(
    combine_embeddings,
    axis=1,
    meta=('embedding', 'object')
)

# Select the item_id and the new embedding column, dropping the old ones
gmf_book_embeddings_final = gmf_book_embeddings_df[['item_id', 'embedding']]

# Merge book embeddings with item ID map
gmf_book_embeddings_final = gmf_book_embeddings_final.merge(
    item_id_map,
    left_on='item_id',
    right_on='new_itemId',
    how='inner'
)

# Select and rename columns
gmf_book_embeddings_final = gmf_book_embeddings_final[['original_itemId', 'embedding']]
gmf_book_embeddings_final = gmf_book_embeddings_final.rename(columns={'original_itemId': 'item_id'})

sbert_embeddings_df = dd.read_parquet("embeddings/sbert_embeddings.parquet")
sbert_embeddings_df.head()

# Setting Up Qdrant
client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="sbert_embeddings",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

client.create_collection(
    collection_name="gmf_user_embeddings",
    vectors_config=VectorParams(size=32, distance=Distance.DOT),
)

client.create_collection(
    collection_name="gmf_book_embeddings",
    vectors_config=VectorParams(size=32, distance=Distance.DOT),
)

batch_size = 500 # Define batch size for uploads

# --- Upload GMF User Embeddings ---
print("Uploading GMF User Embeddings...")
user_points_to_upload = []
# Compute the Dask DataFrame
computed_user_df = gmf_user_embeddings_final.compute()
for index, row in computed_user_df.iterrows():
    # User IDs are strings (hashes) based on previous error
    user_id_val = str(row['user_id']) # Ensure it's treated as a string
    user_points_to_upload.append(PointStruct(
        id=user_id_val, # Use string user_id as the point ID
        vector=row['embedding'],
        payload={"user_id": user_id_val} # Store string user_id in payload
    ))

# Upsert GMF User embeddings in batches
print(f"Upserting {len(user_points_to_upload)} GMF user points in batches of {batch_size}...")
for i in range(0, len(user_points_to_upload), batch_size):
    batch = user_points_to_upload[i:i + batch_size]
    client.upsert(collection_name="gmf_user_embeddings", points=batch, wait=True)
print(f"Uploaded {len(user_points_to_upload)} GMF user points.")
print("-" * 30)


# --- Upload GMF Book Embeddings ---
print("Uploading GMF Book Embeddings...")
book_points_to_upload = []
# Compute the Dask DataFrame
computed_book_df = gmf_book_embeddings_final.compute()
correct_book_id_column_gmf = 'item_id' # Based on previous output

for index, row in computed_book_df.iterrows():
    try:
        # Assume item_id is integer, handle potential errors
        point_id = int(row[correct_book_id_column_gmf])
        payload_id = point_id
    except ValueError:
        print(f"Warning: Could not convert ID '{row[correct_book_id_column_gmf]}' to int for GMF book embedding. Using as string.")
        point_id = str(row[correct_book_id_column_gmf])
        payload_id = point_id

    book_points_to_upload.append(PointStruct(
        id=point_id, # Use original item_id as the point ID (int or string)
        vector=row['embedding'],
        payload={correct_book_id_column_gmf: payload_id} # Store item_id in payload
    ))

# Upsert GMF Book embeddings in batches
print(f"Upserting {len(book_points_to_upload)} GMF book points in batches of {batch_size}...")
for i in range(0, len(book_points_to_upload), batch_size):
    batch = book_points_to_upload[i:i + batch_size]
    client.upsert(collection_name="gmf_book_embeddings", points=batch, wait=True)
print(f"Uploaded {len(book_points_to_upload)} GMF book points.")
print("-" * 30)


# --- Upload SBERT Book Embeddings ---
print("Uploading SBERT Book Embeddings...")
sbert_points_to_upload = []
# Compute the Dask DataFrame
computed_sbert_df = sbert_embeddings_df.compute()
correct_book_id_column_sbert = 'book_id' # Based on previous output

for index, row in computed_sbert_df.iterrows():
    try:
        # Assume book_id is integer, handle potential errors
        point_id = int(row[correct_book_id_column_sbert])
        payload_id = point_id
    except ValueError:
        print(f"Warning: Could not convert ID '{row[correct_book_id_column_sbert]}' to int for SBERT embedding. Using as string.")
        point_id = str(row[correct_book_id_column_sbert])
        payload_id = point_id

    sbert_points_to_upload.append(PointStruct(
        id=point_id, # Use the determined point ID (int or string)
        vector=row['embedding'], # Assuming 'embedding' column is correct
        payload={
            correct_book_id_column_sbert: payload_id,
            "text": row.get("text", "") # Include 'text' in payload, handle if missing
        }
    ))

# Upsert SBERT embeddings in batches
print(f"Upserting {len(sbert_points_to_upload)} SBERT book points in batches of {batch_size}...")
for i in range(0, len(sbert_points_to_upload), batch_size):
    batch = sbert_points_to_upload[i:i + batch_size]
    client.upsert(collection_name="sbert_embeddings", points=batch, wait=True)
print(f"Uploaded {len(sbert_points_to_upload)} SBERT book points.")
print("-" * 30)

print("All uploads complete.")