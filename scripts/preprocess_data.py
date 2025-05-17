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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import boto3
from botocore.config import Config

load_dotenv()


# Load Data

# Books
#books_df = pd.read_json("data/goodreads_books.json", lines=True)
#books_df = pd.read_pickle("data/books_df.pkl")
#books_df.to_parquet("data/books_df.parquet")
#books_df = pd.read_pickle("data/books_df.pkl")
#books_df = pd.read_parquet("../data/books_df.parquet")

# print("\n\nAuthors for first 5 records:")
# print(books_df['authors'].head())



# Interactions CSV
#interactions_df = pd.read_csv("data/goodreads_interactions.csv")
#interactions_df.to_pickle("data/interactions_df.pkl")
#interactions_df = pd.read_pickle("data/interactions_df.pkl")


book_id_map_df = pd.read_csv("../data/book_id_map.csv")
user_id_map_df = pd.read_csv("../data/user_id_map.csv")

# Map user_id_csv to actual user_id
interactions_df['user_id'] = interactions_df['user_id'].map(user_id_map_df.set_index('user_id_csv')['user_id'])

# # Map book_id to actual book_id
interactions_df['book_id'] = interactions_df['book_id'].map(book_id_map_df.set_index('book_id_csv')['book_id'])

# %%
interactions_df.to_parquet("data/interactions_df.parquet")

# %%
interactions_df.head()

# %%
len(interactions_df)

# %% [markdown]
# ## Interactions Dedup JSON

# %%
# # Process JSON in chunks and save to parquet, tracking progress
# import pyarrow.parquet as pq
# import pyarrow as pa

# chunk_size = 150000
# output_path = "data/interactions_dedup.parquet"
# progress_path = "data/chunk_progress.txt"

# # Get starting chunk from progress file if it exists
# start_chunk = 0
# if os.path.exists(progress_path):
#     with open(progress_path) as f:
#         start_chunk = int(f.read())

# # Create ParquetWriter for appending
# writer = None

# for chunk_count, chunk in enumerate(pd.read_json("data/goodreads_interactions_dedup.json", lines=True, chunksize=chunk_size)):
#     if chunk_count < start_chunk:
#         continue
        
#     table = pa.Table.from_pandas(chunk)
    
#     if writer is None:
#         writer = pq.ParquetWriter(output_path, table.schema)
    
#     writer.write_table(table)
    
#     # Save progress
#     with open(progress_path, 'w') as f:
#         f.write(str(chunk_count + 1))

# # Close the writer when done
# if writer:
#     writer.close()



# Reviews
# reviews_df = pd.read_json("data/goodreads_reviews.json", lines=True)
# reviews_df.to_parquet("data/reviews_df.parquet")
# reviews_df = pd.read_parquet("data/reviews_df.parquet")
# reviews_df.head()

# Books Works
# books_works_df = pd.read_json("data/goodreads_book_works.json", lines=True)
# books_works_df.to_parquet("data/books_works_df.parquet")
# books_works_df.head()


# Read the gzipped JSON file
#authors_df = pd.read_json("data/authors.json", lines=True)
#authors_df.to_pickle("data/authors_df.pkl")
# Display the first few rows
#authors_df.head()