import pandas as pd
import numpy as np
import dask.dataframe as dd
import os

class DataReducer:
    """
    Reduces datasets based on interaction counts and ratings.

    Filters interactions, users, books, and reviews based on specified thresholds
    and saves the reduced datasets.
    """
    def __init__(self, data_dir="../data", output_dir="../data/reduced",
                 min_rating=4, min_user_interactions=100, min_book_interactions=500):
        """
        Initializes the DataReducer.

        Args:
            data_dir (str): Directory containing the input parquet/CSV files.
            output_dir (str): Directory to save the reduced output files.
            min_rating (int): Minimum rating to keep an interaction.
            min_user_interactions (int): Minimum interactions for a user to be kept.
            min_book_interactions (int): Minimum interactions for a book to be kept.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.min_rating = min_rating
        self.min_user_interactions = min_user_interactions
        self.min_book_interactions = min_book_interactions

        self.interactions_dedup_path = os.path.join(self.data_dir, "interactions_dedup.parquet")
        self.books_path = os.path.join(self.data_dir, "new_books.parquet")
        self.reviews_path = os.path.join(self.data_dir, "reviews_dedup.parquet")
        # self.books_works_path = os.path.join(self.data_dir, "books_works_df.parquet") # Not used in reduction logic
        # self.user_id_map_path = os.path.join(self.data_dir, "user_id_map.csv") # Not used in reduction logic
        # self.book_id_map_path = os.path.join(self.data_dir, "book_id_map.csv") # Not used in reduction logic

        self.reduced_user_ids_path = os.path.join(self.output_dir, "reduced_user_ids.csv")
        self.reduced_book_ids_path = os.path.join(self.output_dir, "reduced_book_ids.csv")
        self.reduced_interactions_path = os.path.join(self.output_dir, "reduced_interactions.parquet")
        self.reduced_books_path = os.path.join(self.output_dir, "reduced_books.parquet")
        self.reduced_reviews_path = os.path.join(self.output_dir, "reduced_reviews.parquet")

        self.valid_users = None
        self.valid_books = None
        self.interactions_filtered_rating_df = None # Can be Dask or Pandas

        os.makedirs(self.output_dir, exist_ok=True)
        self._configure_pandas()

    def _configure_pandas(self):
        """Sets pandas display options."""
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

    def _load_interactions(self):
        """Loads the deduplicated interactions data."""
        print(f"Loading interactions from {self.interactions_dedup_path}...")
        return dd.read_parquet(self.interactions_dedup_path)

    def _filter_interactions_by_rating(self, interactions_df):
        """Filters interactions based on the minimum rating."""
        print(f"Filtering interactions with rating >= {self.min_rating}...")
        self.interactions_filtered_rating_df = interactions_df[interactions_df['rating'] >= self.min_rating]
        # Keep as Dask DataFrame for now

    def _calculate_and_filter_ids(self):
        """Calculates interaction counts and filters users and books."""
        if self.interactions_filtered_rating_df is None:
            raise ValueError("Rating-filtered interactions not available. Run _filter_interactions_by_rating first.")

        print("Calculating user interaction counts...")
        user_counts = self.interactions_filtered_rating_df.groupby('user_id').size().compute()
        self.valid_users = user_counts[user_counts >= self.min_user_interactions].index
        print(f"Found {len(self.valid_users)} valid users with >= {self.min_user_interactions} interactions.")

        print("Calculating book interaction counts...")
        book_counts = self.interactions_filtered_rating_df.groupby('book_id').size().compute()
        self.valid_books = book_counts[book_counts >= self.min_book_interactions].index
        print(f"Found {len(self.valid_books)} valid books with >= {self.min_book_interactions} interactions.")

    def _save_valid_ids(self):
        """Saves the valid user and book IDs to CSV files."""
        if self.valid_users is None or self.valid_books is None:
            raise ValueError("Valid user/book IDs not calculated yet.")

        print(f"Saving valid user IDs to {self.reduced_user_ids_path}...")
        pd.Series(self.valid_users, name='user_id').to_csv(self.reduced_user_ids_path, index=False)

        print(f"Saving valid book IDs to {self.reduced_book_ids_path}...")
        pd.Series(self.valid_books, name='book_id').to_csv(self.reduced_book_ids_path, index=False)

    def _reduce_interactions(self):
        """Reduces the interactions dataframe to include only valid users and books."""
        if self.valid_users is None or self.valid_books is None:
            raise ValueError("Valid user/book IDs not available.")
        if self.interactions_filtered_rating_df is None:
             raise ValueError("Rating-filtered interactions not available.")

        print("Reducing interactions dataframe...")
        # Compute the Dask DataFrame before filtering with Pandas Series/Index
        interactions_pd = self.interactions_filtered_rating_df.compute()

        interactions_filtered_df = interactions_pd[
            interactions_pd['user_id'].isin(self.valid_users) &
            interactions_pd['book_id'].isin(self.valid_books)
        ]
        print(f"Reduced interactions dataframe shape: {interactions_filtered_df.shape}")
        print(f"Saving reduced interactions to {self.reduced_interactions_path}...")
        interactions_filtered_df.to_parquet(self.reduced_interactions_path, index=False)

    def _filter_similar_books_array(self, arr, valid_book_ids_set):
        """Helper function to filter the 'similar_books' numpy array."""
        if isinstance(arr, np.ndarray) and arr.size > 0:
            # Ensure comparison is done with correct types (assuming IDs in array are strings)
            try:
                return np.array([int(book_id) for book_id in arr if int(book_id) in valid_book_ids_set], dtype=int)
            except ValueError: # Handle cases where conversion might fail
                 return np.array([], dtype=int)
        return np.array([], dtype=int)

    def _reduce_books(self):
        """Reduces the books dataframe and filters the 'similar_books' column."""
        if self.valid_books is None:
            raise ValueError("Valid book IDs not available.")

        print(f"Loading books data from {self.books_path}...")
        books_df = dd.read_parquet(self.books_path)

        print("Filtering books dataframe...")
        # Ensure valid_books is in a format usable by Dask's isin (like a list or Pandas Series)
        valid_books_list = self.valid_books.tolist()
        filtered_books_ddf = books_df[books_df['book_id'].isin(valid_books_list)]

        print("Computing filtered books dataframe...")
        filtered_books_pdf = filtered_books_ddf.compute()
        print(f"Reduced books dataframe shape (before similar_books filter): {filtered_books_pdf.shape}")

        print("Filtering 'similar_books' column...")
        valid_book_ids_set = set(self.valid_books.astype(int)) # Use a set for efficient lookup
        filtered_books_pdf['similar_books'] = filtered_books_pdf['similar_books'].apply(
            self._filter_similar_books_array, args=(valid_book_ids_set,)
        )

        print(f"Saving reduced books to {self.reduced_books_path}...")
        filtered_books_pdf.to_parquet(self.reduced_books_path, index=False)
        print(filtered_books_pdf[['book_id', 'similar_books']].head())


    def _reduce_reviews(self):
        """Reduces the reviews dataframe."""
        if self.valid_users is None or self.valid_books is None:
            raise ValueError("Valid user/book IDs not available.")

        print(f"Loading reviews data from {self.reviews_path}...")
        reviews_df = dd.read_parquet(self.reviews_path)

        print("Filtering reviews dataframe...")
        # Ensure valid IDs are in a format usable by Dask's isin
        valid_users_list = self.valid_users.tolist()
        valid_books_list = self.valid_books.tolist()

        filtered_reviews_ddf = reviews_df[
            reviews_df['user_id'].isin(valid_users_list) &
            reviews_df['book_id'].isin(valid_books_list)
        ]

        print("Computing filtered reviews dataframe...")
        filtered_reviews_pdf = filtered_reviews_ddf.compute()
        print(f"Reduced reviews dataframe shape: {filtered_reviews_pdf.shape}")

        print(f"Saving reduced reviews to {self.reduced_reviews_path}...")
        filtered_reviews_pdf.to_parquet(self.reduced_reviews_path, index=False)
        print(filtered_reviews_pdf.head())

    def run(self):
        """Executes the full data reduction pipeline."""
        print("Starting data reduction process...")
        interactions_df = self._load_interactions()
        self._filter_interactions_by_rating(interactions_df)
        self._calculate_and_filter_ids()
        self._save_valid_ids()
        self._reduce_interactions() # Depends on computed interactions_filtered_rating_df and valid IDs
        self._reduce_books()      # Depends on valid_books
        self._reduce_reviews()    # Depends on valid_users and valid_books
        print("Data reduction process finished.")

# Example Usage:
if __name__ == "__main__":
    # Define paths relative to the script location or use absolute paths
    current_dir = os.path.dirname(__file__) # Gets the directory where the script is located
    project_root = os.path.abspath(os.path.join(current_dir, '..')) # Assumes script is in 'scripts' subdir
    data_directory = os.path.join(project_root, 'data')
    output_directory = os.path.join(project_root, 'data', 'reduced_oop') # Specify a different output dir if needed

    reducer = DataReducer(
        data_dir=data_directory,
        output_dir=output_directory,
        min_rating=4,
        min_user_interactions=100,
        min_book_interactions=500
    )
    reducer.run()