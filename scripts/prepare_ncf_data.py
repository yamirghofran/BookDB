# Preparing Interactions Data for Neural Collaborative Filtering Training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import dask.dataframe as dd
import torch

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Loading the datasets
interactions_reduced_df = dd.read_parquet("../data/reduced_interactions.parquet")
user_id_map = pd.read_csv("../data/user_id_map.csv")
book_id_map = pd.read_csv("../data/book_id_map.csv")

# Select and rename columns on the filtered Dask DataFrame
interactions_prepared_df = interactions_reduced_df[['user_id', 'book_id', 'rating', 'date_updated']].rename(columns={
    'user_id': 'userId',
    'book_id': 'itemId',
    'date_updated': 'timestamp'
})

# Convert 'timestamp' column using Dask's to_datetime
# Use utc=True to convert parsed times (with offset from %z) directly to UTC
interactions_prepared_df['timestamp'] = dd.to_datetime(
    interactions_prepared_df['timestamp'],
    format='%a %b %d %H:%M:%S %z %Y',
    errors='coerce',
    utc=True  # Add this argument
)

# Convert datetime objects (now timezone-aware UTC) to Unix timestamp
# Casting timezone-aware datetime to int64 gives nanoseconds since UTC epoch
interactions_prepared_df['timestamp'] = (interactions_prepared_df['timestamp'].astype(np.int64) // 10**9)



print("Displaying head using Dask (memory efficient):")
print(interactions_prepared_df.head())

# Create Reduced Mappings
# Compute unique user and item IDs from the prepared dataframe
unique_users = interactions_prepared_df['userId'].unique().compute()
unique_items = interactions_prepared_df['itemId'].unique().compute()

# Create mappings from original IDs to new consecutive integer IDs starting from 0
user_map = pd.Series(range(len(unique_users)), index=unique_users)
item_map = pd.Series(range(len(unique_items)), index=unique_items)

# Save the user mapping to CSV
user_map_df = user_map.reset_index()
user_map_df.columns = ['original_userId', 'new_userId']
user_map_df.to_csv("data/user_id_map_reduced.csv", index=False)
print(f"Reduced user ID mapping saved to data/user_id_map_reduced.csv. Total users: {len(user_map_df)}")

# Save the item mapping to CSV
item_map_df = item_map.reset_index()
item_map_df.columns = ['original_itemId', 'new_itemId']
item_map_df.to_csv("data/item_id_map_reduced.csv", index=False)
print(f"Reduced item ID mapping saved to data/item_id_map_reduced.csv. Total items: {len(item_map_df)}")

# Apply the mappings to the Dask DataFrame
# Use .map() which works efficiently with a Pandas Series map
interactions_final_df = interactions_prepared_df.copy()
interactions_final_df['userId'] = interactions_final_df['userId'].map(user_map, meta=('userId', 'int64'))
interactions_final_df['itemId'] = interactions_final_df['itemId'].map(item_map, meta=('itemId', 'int64'))

# Verify the result (optional)
print("\nDisplaying head of final DataFrame with new integer IDs:")
print(interactions_final_df.head())

# Save to Parquet
# Define the output filename (ensure it ends with .parquet)
output_filename = "../data/interactions_prepared_ncf_reduced.parquet"

# Repartition to a single partition and save to a single file
# Use write_index=False instead of index=False
interactions_final_df.repartition(npartitions=1).to_parquet(
    output_filename,
    write_index=False
)

print(f"Prepared interactions DataFrame saved to single file: {output_filename}")


