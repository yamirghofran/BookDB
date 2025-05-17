import pandas as pd
import uuid
import dask.dataframe as dd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


authors_df = pd.read_parquet("../data/authors_df.parquet")
authors_df.head()

# Create a dictionary mapping old author_ids to new UUIDs
author_id_map = {old_id: str(uuid.uuid4()) for old_id in authors_df['author_id'].unique()}

# Create a copy and replace the IDs
new_authors_df = authors_df.copy()
new_authors_df['author_id'] = new_authors_df['author_id'].map(author_id_map)

# Save the mapping for later use
author_id_map_df = pd.DataFrame(list(author_id_map.items()), columns=['old_id', 'new_id'])
author_id_map_df.to_csv("data/author_id_map.csv", index=False)

authors_df = new_authors_df

new_authors_df.to_parquet("data/new_authors.parquet")
new_authors_df.head()

# Changing author_id in books
books_df = pd.read_parquet("../data/books_df.parquet")
books_df.head()

# Load the author ID mapping and ensure index is string type
author_id_map = pd.read_csv("../data/author_id_map.csv")
author_id_map['old_id'] = author_id_map['old_id'].astype(str)  # Convert to string
author_id_map = author_id_map.set_index('old_id')['new_id']

new_books_df = books_df.copy()

# Convert the list of dicts to just a list of author UUIDs
new_books_df['authors'] = new_books_df['authors'].apply(
    lambda x: [author_id_map[str(author['author_id'])] for author in x]
)

# Save updated books dataframe
new_books_df.to_parquet("data/new_books.parquet")
new_books_df.head()


