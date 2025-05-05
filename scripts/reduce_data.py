
# Reducing the data
# There are 238 million interactions and 2.3 million books in this dataset. For our purposes of training and tuning Neural Collaborative Filtering and Transformer models, this size is impossible to manage in terms of memory and compute.

# Therefore, we reduce the dataset to only include interactions with explicit ratings of 4 and 5, users with 100+ interactions, and books with 500+ interactions.

# 1. Configuration and Setup
import pandas as pd
import numpy as np
import dask.dataframe as dd
import pyarrow.parquet as pq

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Load the original datasets
interactions_dedup_df = dd.read_parquet("../data/interactions_dedup.parquet")
books_df = dd.read_parquet("../data/new_books.parquet")
books_works_df = dd.read_parquet("../data/books_works_df.parquet")
user_id_map = pd.read_csv("../data/user_id_map.csv")
book_id_map = pd.read_csv("../data/book_id_map.csv")

# 2. Filtering based on interactions
# Filter interactions with rating >= 4
interactions_filtered_rating_df = interactions_dedup_df[interactions_dedup_df['rating'] >= 4]
interactions_filtered_rating_df = interactions_filtered_rating_df

# Calculate user interaction counts
user_counts = interactions_filtered_rating_df.groupby('user_id').size().compute()

# Filter users with 100+ interactions
valid_users = user_counts[user_counts >= 100].index

# Calculate book interaction counts
book_counts = interactions_filtered_rating_df.groupby('book_id').size().compute()
# Filter books with 500+ interactions
valid_books = book_counts[book_counts >= 500].index

# Save valid user IDs to a CSV file
pd.Series(valid_users, name='user_id').to_csv("../data/reduced_user_ids.csv", index=False)
print("Valid user IDs saved to data/reduced_user_ids.csv")

# Save valid book IDs to a CSV file
pd.Series(valid_books, name='book_id').to_csv("../data/reduced_book_ids.csv", index=False)
print("Valid book IDs saved to data/reduced_book_ids.csv")

interactions_filtered_rating_df = interactions_filtered_rating_df.compute()

# 3. Reducing Interactions Dataframe
interactions_filtered_df = interactions_filtered_rating_df[
    interactions_filtered_rating_df['user_id'].isin(valid_users) &
    interactions_filtered_rating_df['book_id'].isin(valid_books)
]

interactions_filtered_df.to_parquet("data/reduced_interactions.parquet", index=False)

# 4. Reducing Books Dataframe
reduced_book_ids = pd.read_csv("../data/reduced_book_ids.csv")
# Get the list of book IDs to keep
book_ids_to_keep = reduced_book_ids['book_id'].unique()

# Filter the Dask DataFrame
filtered_books_df = books_df[books_df['book_id'].isin(book_ids_to_keep)]

# Optionally, trigger computation and view the head
filtered_books_df = filtered_books_df.compute()
len(filtered_books_df)

# Filtering the `similar_books` attribute

# Convert book_ids_to_keep to a set for efficient lookup
# Ensure the IDs in the set are integers, matching the type in reduced_book_ids
book_ids_to_keep_set = set(book_ids_to_keep.astype(int))

# Define a function to filter the similar_books array
def filter_array(arr):
    # Check if the input is a numpy array and not empty
    if isinstance(arr, np.ndarray) and arr.size > 0:
        # Convert string IDs in the array to integers for comparison
        # Keep only those IDs present in the book_ids_to_keep_set
        return np.array([book_id for book_id in arr if int(book_id) in book_ids_to_keep_set], dtype=object)
    # Return an empty numpy array if input is not valid or empty
    return np.array([], dtype=object)

# Apply the filtering function to the 'similar_books' column
filtered_books_df['similar_books'] = filtered_books_df['similar_books'].apply(filter_array)

# Display the head to verify the changes (optional)
print(filtered_books_df[['book_id', 'similar_books']].head())
filtered_books_df.to_parquet("data/reduced_books.parquet")

# 5. Reducing Reviews Dataframe
reviews_df = dd.read_parquet("../data/reviews_dedup.parquet")
reduced_user_ids = pd.read_csv("../data/reduced_user_ids.csv")
user_ids_to_keep = reduced_user_ids['user_id'].unique()

# Filter reviews_df
filtered_reviews_df = reviews_df[
    reviews_df['user_id'].isin(user_ids_to_keep) &
    reviews_df['book_id'].isin(book_ids_to_keep)
]

pd_reviews_df = filtered_reviews_df.compute()
pd_reviews_df.to_parquet("data/reduced_reviews.parquet")

print(len(pd_reviews_df))
print(pd_reviews_df.head())


