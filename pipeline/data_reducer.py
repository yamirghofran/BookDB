import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import datetime
from typing import Dict, Any
from .core import PipelineStep
from utils import send_discord_webhook

class DataReducerStep(PipelineStep):
    """
    Reduces datasets based on interaction counts and ratings.
    Filters interactions, users, books, and reviews based on specified thresholds
    and saves the reduced datasets.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.data_dir = "data"
        self.output_dir = "data"
        self.min_rating = 4
        self.min_user_interactions = 100
        self.min_book_interactions = 500
        self.interactions_dedup_path = None
        self.books_path = None
        self.reviews_path = None
        self.reduced_user_ids_path = None
        self.reduced_book_ids_path = None
        self.reduced_interactions_path = None
        self.reduced_books_path = None
        self.reduced_reviews_path = None
        self.valid_users = None
        self.valid_books = None
        self.interactions_filtered_rating_df = None

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self.data_dir = self.config.get("data_dir", self.data_dir)
        self.output_dir = self.config.get("output_dir", self.output_dir)
        self.min_rating = self.config.get("min_rating", self.min_rating)
        self.min_user_interactions = self.config.get("min_user_interactions", self.min_user_interactions)
        self.min_book_interactions = self.config.get("min_book_interactions", self.min_book_interactions)
        self.interactions_dedup_path = os.path.join(self.data_dir, self.config.get("interactions_dedup_file", "interactions_dedup.parquet"))
        self.books_path = os.path.join(self.data_dir, self.config.get("books_file", "new_books.parquet"))
        self.reviews_path = os.path.join(self.data_dir, self.config.get("reviews_file", "reviews_dedup.parquet"))
        self.reduced_user_ids_path = os.path.join(self.output_dir, self.config.get("reduced_user_ids_file", "reduced_user_ids.csv"))
        self.reduced_book_ids_path = os.path.join(self.output_dir, self.config.get("reduced_book_ids_file", "reduced_book_ids.csv"))
        self.reduced_interactions_path = os.path.join(self.output_dir, self.config.get("reduced_interactions_file", "reduced_interactions.parquet"))
        self.reduced_books_path = os.path.join(self.output_dir, self.config.get("reduced_books_file", "reduced_books.parquet"))
        self.reduced_reviews_path = os.path.join(self.output_dir, self.config.get("reduced_reviews_file", "reduced_reviews.parquet"))
        os.makedirs(self.output_dir, exist_ok=True)
        self._configure_pandas()

    def _configure_pandas(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

    def _send_notification(self, title: str, description: str, color: int = 0x00FF00, fields: list = None, error: bool = False):
        """Send a Discord notification with consistent formatting."""
        try:
            embed = {
                "title": f"📉 {title}" if not error else f"❌ {title}",
                "description": description,
                "color": color if not error else 0xFF0000,  # Red for errors
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "footer": {"text": f"Pipeline Step: {self.name}"}
            }
            
            if fields:
                embed["fields"] = fields
                
            send_discord_webhook(
                content=None,
                embed=embed,
                username="BookDB Pipeline"
            )
        except Exception as e:
            self.logger.warning(f"Failed to send Discord notification: {e}")

    def _load_interactions(self):
        self.logger.info(f"Loading interactions from {self.interactions_dedup_path}...")
        return dd.read_parquet(self.interactions_dedup_path)

    def _filter_interactions_by_rating(self, interactions_df):
        self.logger.info(f"Filtering interactions with rating >= {self.min_rating}...")
        self.interactions_filtered_rating_df = interactions_df[interactions_df['rating'] >= self.min_rating]

    def _calculate_and_filter_ids(self):
        if self.interactions_filtered_rating_df is None:
            raise ValueError("Rating-filtered interactions not available. Run _filter_interactions_by_rating first.")
        try:
            self.logger.info("Calculating user interaction counts...")
            user_counts = self.interactions_filtered_rating_df.groupby('user_id').size().compute()
            self.valid_users = user_counts[user_counts >= self.min_user_interactions].index
            self.logger.info(f"Found {len(self.valid_users)} valid users with >= {self.min_user_interactions} interactions.")
            
            self.logger.info("Calculating book interaction counts...")
            book_counts = self.interactions_filtered_rating_df.groupby('book_id').size().compute()
            self.valid_books = book_counts[book_counts >= self.min_book_interactions].index
            self.logger.info(f"Found {len(self.valid_books)} valid books with >= {self.min_book_interactions} interactions.")
            
            # Send filtering results notification
            total_users = len(user_counts)
            total_books = len(book_counts)
            self._send_notification(
                "Data Filtering Complete",
                f"Calculated interaction counts and applied filtering thresholds",
                fields=[
                    {"name": "Total Users", "value": f"{total_users:,}", "inline": True},
                    {"name": "Valid Users", "value": f"{len(self.valid_users):,}", "inline": True},
                    {"name": "User Retention", "value": f"{len(self.valid_users)/total_users*100:.1f}%", "inline": True},
                    {"name": "Total Books", "value": f"{total_books:,}", "inline": True},
                    {"name": "Valid Books", "value": f"{len(self.valid_books):,}", "inline": True},
                    {"name": "Book Retention", "value": f"{len(self.valid_books)/total_books*100:.1f}%", "inline": True},
                    {"name": "Min User Interactions", "value": f"{self.min_user_interactions}", "inline": True},
                    {"name": "Min Book Interactions", "value": f"{self.min_book_interactions}", "inline": True}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to calculate and filter IDs: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Data Filtering Failed", 
                error_msg,
                error=True
            )
            raise

    def _save_valid_ids(self):
        if self.valid_users is None or self.valid_books is None:
            raise ValueError("Valid user/book IDs not calculated yet.")
        self.logger.info(f"Saving valid user IDs to {self.reduced_user_ids_path}...")
        pd.Series(self.valid_users, name='user_id').to_csv(self.reduced_user_ids_path, index=False)
        self.logger.info(f"Saving valid book IDs to {self.reduced_book_ids_path}...")
        pd.Series(self.valid_books, name='book_id').to_csv(self.reduced_book_ids_path, index=False)

    def _reduce_interactions(self):
        if self.valid_users is None or self.valid_books is None:
            raise ValueError("Valid user/book IDs not available.")
        if self.interactions_filtered_rating_df is None:
             raise ValueError("Rating-filtered interactions not available.")
        try:
            self.logger.info("Reducing interactions dataframe...")
            interactions_pd = self.interactions_filtered_rating_df.compute()
            original_count = len(interactions_pd)
            interactions_filtered_df = interactions_pd[
                interactions_pd['user_id'].isin(self.valid_users) &
                interactions_pd['book_id'].isin(self.valid_books)
            ]
            final_count = len(interactions_filtered_df)
            self.logger.info(f"Reduced interactions dataframe shape: {interactions_filtered_df.shape}")
            self.logger.info(f"Saving reduced interactions to {self.reduced_interactions_path}...")
            interactions_filtered_df.to_parquet(self.reduced_interactions_path, index=False)
            
            # Send success notification
            self._send_notification(
                "Interactions Reduction Complete",
                f"Successfully reduced interactions dataset",
                fields=[
                    {"name": "Original Interactions", "value": f"{original_count:,}", "inline": True},
                    {"name": "Reduced Interactions", "value": f"{final_count:,}", "inline": True},
                    {"name": "Retention Rate", "value": f"{final_count/original_count*100:.1f}%", "inline": True},
                    {"name": "Output File", "value": f"`{os.path.basename(self.reduced_interactions_path)}`", "inline": False}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to reduce interactions: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Interactions Reduction Failed",
                error_msg,
                error=True
            )
            raise

    def _filter_similar_books_array(self, arr, valid_book_ids_set):
        if isinstance(arr, np.ndarray) and arr.size > 0:
            try:
                return np.array([int(book_id) for book_id in arr if int(book_id) in valid_book_ids_set], dtype=int)
            except ValueError:
                 return np.array([], dtype=int)
        return np.array([], dtype=int)

    def _reduce_books(self):
        if self.valid_books is None:
            raise ValueError("Valid book IDs not available.")
        try:
            self.logger.info(f"Loading books data from {self.books_path}...")
            books_df = dd.read_parquet(self.books_path)
            self.logger.info("Filtering books dataframe...")
            valid_books_list = self.valid_books.tolist()
            filtered_books_ddf = books_df[books_df['book_id'].isin(valid_books_list)]
            self.logger.info("Computing filtered books dataframe...")
            filtered_books_pdf = filtered_books_ddf.compute()
            original_count = len(books_df.compute())
            self.logger.info(f"Reduced books dataframe shape (before similar_books filter): {filtered_books_pdf.shape}")
            self.logger.info("Filtering 'similar_books' column...")
            valid_book_ids_set = set(self.valid_books.astype(int))
            filtered_books_pdf['similar_books'] = filtered_books_pdf['similar_books'].apply(
                self._filter_similar_books_array, args=(valid_book_ids_set,)
            )
            final_count = len(filtered_books_pdf)
            self.logger.info(f"Saving reduced books to {self.reduced_books_path}...")
            filtered_books_pdf.to_parquet(self.reduced_books_path, index=False)
            
            # Send success notification
            self._send_notification(
                "Books Reduction Complete",
                f"Successfully reduced books dataset and filtered similar_books references",
                fields=[
                    {"name": "Original Books", "value": f"{original_count:,}", "inline": True},
                    {"name": "Reduced Books", "value": f"{final_count:,}", "inline": True},
                    {"name": "Retention Rate", "value": f"{final_count/original_count*100:.1f}%", "inline": True},
                    {"name": "Output File", "value": f"`{os.path.basename(self.reduced_books_path)}`", "inline": False}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to reduce books: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Books Reduction Failed",
                error_msg,
                error=True
            )
            raise

    def _reduce_reviews(self):
        if self.valid_users is None or self.valid_books is None:
            raise ValueError("Valid user/book IDs not available.")
        try:
            self.logger.info(f"Loading reviews data from {self.reviews_path}...")
            reviews_df = dd.read_parquet(self.reviews_path)
            original_count = len(reviews_df.compute())
            self.logger.info("Filtering reviews dataframe...")
            valid_users_list = self.valid_users.tolist()
            valid_books_list = self.valid_books.tolist()
            filtered_reviews_ddf = reviews_df[
                reviews_df['user_id'].isin(valid_users_list) &
                reviews_df['book_id'].isin(valid_books_list)
            ]
            self.logger.info("Computing filtered reviews dataframe...")
            filtered_reviews_pdf = filtered_reviews_ddf.compute()
            final_count = len(filtered_reviews_pdf)
            self.logger.info(f"Reduced reviews dataframe shape: {filtered_reviews_pdf.shape}")
            self.logger.info(f"Saving reduced reviews to {self.reduced_reviews_path}...")
            filtered_reviews_pdf.to_parquet(self.reduced_reviews_path, index=False)
            
            # Send success notification
            self._send_notification(
                "Reviews Reduction Complete",
                f"Successfully reduced reviews dataset",
                fields=[
                    {"name": "Original Reviews", "value": f"{original_count:,}", "inline": True},
                    {"name": "Reduced Reviews", "value": f"{final_count:,}", "inline": True},
                    {"name": "Retention Rate", "value": f"{final_count/original_count*100:.1f}%", "inline": True},
                    {"name": "Output File", "value": f"`{os.path.basename(self.reduced_reviews_path)}`", "inline": False}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to reduce reviews: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Reviews Reduction Failed",
                error_msg,
                error=True
            )
            raise

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting data reduction process...")
        
        # Send pipeline start notification
        self._send_notification(
            "Data Reduction Started",
            f"Beginning data reduction pipeline: **{self.name}**",
            color=0x0099FF,  # Blue for start
            fields=[
                {"name": "Data Directory", "value": f"`{self.data_dir}`", "inline": True},
                {"name": "Output Directory", "value": f"`{self.output_dir}`", "inline": True},
                {"name": "Min Rating", "value": f"{self.min_rating}", "inline": True},
                {"name": "Min User Interactions", "value": f"{self.min_user_interactions}", "inline": True},
                {"name": "Min Book Interactions", "value": f"{self.min_book_interactions}", "inline": True}
            ]
        )
        
        try:
            outputs = {}
            interactions_df = self._load_interactions()
            self._filter_interactions_by_rating(interactions_df)
            self._calculate_and_filter_ids()
            self._save_valid_ids()
            self._reduce_interactions()
            self._reduce_books()
            self._reduce_reviews()
            outputs["reduced_user_ids_csv"] = self.reduced_user_ids_path
            outputs["reduced_book_ids_csv"] = self.reduced_book_ids_path
            outputs["reduced_interactions_parquet"] = self.reduced_interactions_path
            outputs["reduced_books_parquet"] = self.reduced_books_path
            outputs["reduced_reviews_parquet"] = self.reduced_reviews_path
            self.logger.info("Data reduction process finished.")
            
            # Send pipeline completion notification
            self._send_notification(
                "Data Reduction Complete! 🎉",
                f"All data reduction tasks completed successfully for pipeline: **{self.name}**",
                color=0x00FF00,  # Green for success
                fields=[
                    {"name": "Data Filtering", "value": "✅ Complete", "inline": True},
                    {"name": "Interactions", "value": "✅ Complete", "inline": True},
                    {"name": "Books", "value": "✅ Complete", "inline": True},
                    {"name": "Reviews", "value": "✅ Complete", "inline": True},
                    {"name": "Valid Users", "value": f"{len(self.valid_users):,}", "inline": True},
                    {"name": "Valid Books", "value": f"{len(self.valid_books):,}", "inline": True},
                    {"name": "Output Directory", "value": f"`{self.output_dir}`", "inline": False}
                ]
            )
            
            # Free memory after processing
            self._free_memory(interactions_df, self.interactions_filtered_rating_df, self.valid_users, self.valid_books)
            
            self.output_data = outputs
            return outputs
            
        except Exception as e:
            error_msg = f"Data reduction pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Data Reduction Pipeline Failed",
                error_msg,
                error=True
            )
            # Free memory even on error
            self._free_memory(self.interactions_filtered_rating_df, self.valid_users, self.valid_books)
            raise