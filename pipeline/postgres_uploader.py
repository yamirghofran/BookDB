import os
import pandas as pd
import psycopg2
import psycopg2.extras as extras
import uuid
from faker import Faker
import re
import logging
from datetime import datetime
import pytz # Required for handling timezones robustly
import ast # For safely evaluating string representations of lists
import numpy as np
from typing import Dict, Any
from .core import PipelineStep
from utils import send_discord_webhook

# --- Logging & Faker Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
fake = Faker()

class PostgresUploaderStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        self.db_config = {}
        self.data_paths = {}
        self.conn = None
        self.cursor = None
        self.users_df = None
        self.authors_df = None
        self.books_df = None
        self.reviews_df = None
        self.interactions_df = None
        self.item_id_map_df = None
        self.author_mapping = {}
        self.book_mapping = {}
        self.genre_mapping = {}
        self.all_input_user_ids = set()
        self.book_ncf_id_map = {}

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self.db_config = self.config.get("db_config", self.db_config)
        self.data_paths = self.config.get("data_paths", self.data_paths)

    def _send_notification(self, title: str, description: str, color: int = 0x00FF00, fields: list = None, error: bool = False):
        """Send a Discord notification with consistent formatting."""
        try:
            embed = {
                "title": f"🗄️ {title}" if not error else f"❌ {title}",
                "description": description,
                "color": color if not error else 0xFF0000,  # Red for errors
                "timestamp": datetime.now().isoformat(),
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
            logging.warning(f"Failed to send Discord notification: {e}")

    # --- Helper Functions (Static or Class Methods if they don't need instance state) ---
    @staticmethod
    def safe_to_int(value, default=None):
        try: return int(value) if pd.notna(value) else default
        except (ValueError, TypeError): return default

    @staticmethod
    def safe_to_float(value, default=None):
        try: return float(value) if pd.notna(value) else default
        except (ValueError, TypeError): return default

    @staticmethod
    def parse_goodreads_date(date_str, default=None):
        if pd.isna(date_str) or not date_str: return default
        try:
            dt_part = date_str[:-10].strip(); tz_part = date_str[-10:].strip(); offset_str = tz_part[:5]; year_str = tz_part[5:].strip()
            naive_dt = datetime.strptime(f"{dt_part} {year_str}", '%a %b %d %H:%M:%S %Y')
            offset_hours = int(offset_str[:3]); offset_minutes = int(offset_str[0] + offset_str[3:])
            tz = pytz.FixedOffset((offset_hours * 60) + offset_minutes); aware_dt = tz.localize(naive_dt)
            return aware_dt
        except (ValueError, TypeError, IndexError) as e:
            logging.warning(f"Could not parse date string '{date_str}': {e}"); return default

    @staticmethod
    def parse_list_string(list_str, default=None):
        default_value = default if default is not None else []
        if isinstance(list_str, str):
            if not list_str.strip(): return default_value
            try:
                result = ast.literal_eval(list_str)
                return result if isinstance(result, list) else default_value
            except (ValueError, SyntaxError, TypeError) as e:
                logging.warning(f"Could not parse list string '{str(list_str)[:100]}...': {e}")
                return default_value
        elif isinstance(list_str, list): return list_str
        elif isinstance(list_str, np.ndarray): return list_str.tolist()
        elif pd.isna(list_str): return default_value
        else:
            logging.warning(f"Unexpected type for parse_list_string: {type(list_str)}. Value: {str(list_str)[:100]}...")
            return default_value

    @staticmethod
    def is_valid_uuid(uuid_to_test, version=4):
        if isinstance(uuid_to_test, uuid.UUID): return True
        try: uuid.UUID(str(uuid_to_test), version=version); return True
        except (ValueError, TypeError, AttributeError): return False

    @staticmethod
    def extract_genres(popular_shelves):
        try:
            if not isinstance(popular_shelves, list) or len(popular_shelves) == 0: return []
            found_genres = set(); genre_keywords = ['action', 'adventure', 'comedy', 'crime', 'mystery', 'textbook', 'children', 'mathematics', 'fantasy', 'historical', 'horror', 'romance', 'satire', 'science fiction', 'scifi', 'speculative fiction', 'thriller', 'western', 'paranormal', 'dystopian', 'urban fantasy', 'contemporary', 'young adult', 'ya', 'middle grade', 'children\'s', 'literary fiction', 'magic realism', 'historical fiction', 'gothic', 'suspense', 'biography', 'memoir', 'nonfiction', 'poetry', 'drama', 'historical romance', 'fantasy romance', 'romantic suspense', 'science fiction romance', 'contemporary romance', 'paranormal romance', 'epic fantasy', 'dark fantasy', 'sword and sorcery', 'steampunk', 'cyberpunk', 'apocalyptic', 'post-apocalyptic', 'alternate history', 'superhero', 'mythology', 'fairy tales', 'folklore', 'war', 'military fiction', 'spy fiction', 'political fiction', 'social science fiction', 'techno-thriller', 'medical thriller', 'legal thriller', 'psychological thriller', 'cozy mystery', 'hardboiled', 'noir', 'coming-of-age', 'lgbtq+', 'christian fiction', 'religious fiction', 'humor', 'travel', 'food', 'cooking', 'health', 'self-help', 'business', 'finance', 'history', 'science', 'technology', 'nature', 'art', 'music', 'philosophy', 'education', 'true crime', 'spiritual', 'anthology', 'short stories', 'plays', 'screenplays', 'graphic novel', 'comics', 'manga', 'erotica', 'new adult', 'chick lit', 'womens fiction', 'sports fiction', 'family saga', ' Regency romance', 'literature']; genre_keywords.sort(key=len, reverse=True)
            ignore_keywords = ['to-read', 'owned', 'hardcover', 'shelfari-favorites', 'series', 'might-read', 'dnf-d', 'hambly-barbara', 'strong-females', 'first-in-series', 'no-thanks-series-collections-boxes', 'entertaining-but-limited', 'kate-own', 'e-book', 'compliation', 'my-books', 'books-i-own-but-have-not-read', 'everything-owned', 'books-to-find', 'i-own-it', 'favorite', 'not-read', 'read-some-day', 'library', 'audiobooks', 'status-borrowed', 'owned-books', 'spec-fic-awd-locus-nom', '01', 'hardbacks', 'paper', 'german', 'hardback', 'physical-scifi-fantasy', 'childhood-favorites', 'bundle-same-author', 'aa-sifi-fantasy', 'ready-to-read', 'bought-on-flee-markets', 'fantasy-general', 'hardcopy', 'box-2', 'unfinished', 'magic', 'duplicates', 'favorites', 'books-i-own', 'fantasy-classic', 'own-hard-copy', 'fantasy-read', 'book-club-edition', 'sci-fi-or-fantasy', 'fiction-fantasy', 'fiction-literature-poetry', 'paused-hiatus', 'status—borrowed', 'recs-fantasy', 'fantasy-scifi', 'omnibus', 'speculative', 'sf--fantasy', 'in-my-home-library', 'fant-myth-para-vamps', 'read-in-my-20s']
            for shelf in popular_shelves:
                if not isinstance(shelf, dict) or 'name' not in shelf or not isinstance(shelf['name'], str): continue
                shelf_name = shelf['name'].lower().strip()
                if any(ignore in shelf_name for ignore in ignore_keywords): continue
                for keyword in genre_keywords:
                    if keyword in shelf_name: found_genres.add(keyword)
            return sorted(list(found_genres))
        except Exception as e:
            logging.error(f"Error in extract_genres function processing data: {str(popular_shelves)[:100]}... - Error: {e}", exc_info=True); return []

    def _connect_db(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            logging.info("Connecting to the PostgreSQL database...")
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = False # Ensure transactions are handled explicitly
            psycopg2.extras.register_uuid()
            self.cursor = self.conn.cursor()
            logging.info("Successfully connected to the database.")
            
            # Send success notification
            self._send_notification(
                "Database Connection Established",
                f"Successfully connected to PostgreSQL database",
                fields=[
                    {"name": "Host", "value": f"`{self.db_config.get('host', 'localhost')}`", "inline": True},
                    {"name": "Database", "value": f"`{self.db_config.get('database', 'unknown')}`", "inline": True},
                    {"name": "User", "value": f"`{self.db_config.get('user', 'unknown')}`", "inline": True},
                    {"name": "Auto-commit", "value": "Disabled (transaction mode)", "inline": True},
                    {"name": "Status", "value": "✅ Ready for data upload", "inline": False}
                ]
            )
        except psycopg2.OperationalError as db_err:
            error_msg = f"Database connection failed: {db_err}"
            logging.error(error_msg)
            self._send_notification(
                "Database Connection Failed",
                error_msg,
                error=True,
                fields=[
                    {"name": "Host", "value": f"`{self.db_config.get('host', 'localhost')}`", "inline": True},
                    {"name": "Database", "value": f"`{self.db_config.get('database', 'unknown')}`", "inline": True},
                    {"name": "Error Type", "value": "Operational Error", "inline": True}
                ]
            )
            raise  # Re-raise the exception to be handled by the caller
        except Exception as e:
            error_msg = f"An unexpected error occurred during DB connection: {e}"
            logging.error(error_msg)
            self._send_notification(
                "Database Connection Failed",
                error_msg,
                error=True
            )
            raise

    def _close_db(self):
        """Closes the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
            
    def _load_data(self):
        """Loads all necessary dataframes from the specified file paths."""
        logging.info("Loading data files...")
        try:
            self.books_df = pd.read_parquet(self.data_paths['books'])
            self.interactions_df = pd.read_parquet(self.data_paths['interactions'])
            self.reviews_df = pd.read_parquet(self.data_paths['reviews'])
            self.authors_df = pd.read_parquet(self.data_paths['authors'])
            self.users_df = pd.read_csv(self.data_paths['users'])
            self.item_id_map_df = pd.read_csv(self.data_paths['item_id_map'])
            logging.info("All data files loaded successfully.")
            
            # Send success notification with data statistics
            self._send_notification(
                "Data Loading Complete",
                f"Successfully loaded all data files for database upload",
                fields=[
                    {"name": "Books", "value": f"{len(self.books_df):,} records", "inline": True},
                    {"name": "Interactions", "value": f"{len(self.interactions_df):,} records", "inline": True},
                    {"name": "Reviews", "value": f"{len(self.reviews_df):,} records", "inline": True},
                    {"name": "Authors", "value": f"{len(self.authors_df):,} records", "inline": True},
                    {"name": "Users", "value": f"{len(self.users_df):,} records", "inline": True},
                    {"name": "Item ID Map", "value": f"{len(self.item_id_map_df):,} mappings", "inline": True},
                    {"name": "Total Records", "value": f"{len(self.books_df) + len(self.interactions_df) + len(self.reviews_df) + len(self.authors_df) + len(self.users_df):,}", "inline": False}
                ]
            )
        except FileNotFoundError as e:
            error_msg = f"Error loading data file: {e}. Check paths in data_paths config."
            logging.error(error_msg)
            self._send_notification(
                "Data Loading Failed",
                error_msg,
                error=True,
                fields=[
                    {"name": "Missing File", "value": f"`{str(e).split()[-1] if str(e) else 'Unknown'}`", "inline": True},
                    {"name": "Config Check", "value": "Verify data_paths configuration", "inline": True}
                ]
            )
            raise
        except Exception as e:
            error_msg = f"An error occurred during data loading: {e}"
            logging.error(error_msg)
            self._send_notification(
                "Data Loading Failed",
                error_msg,
                error=True
            )
            raise
        
        # Initialize book_ncf_id_map
        if self.item_id_map_df is not None:
            self.book_ncf_id_map = dict(zip(self.item_id_map_df['itemId'], self.item_id_map_df['ncf_itemId']))
        
        # Populate all_input_user_ids
        if self.users_df is not None:
            try:
                self.all_input_user_ids = set(self.users_df['userId'].astype(str).unique())
                logging.info(f"Collected {len(self.all_input_user_ids)} unique user IDs from input users_df.")
            except KeyError:
                logging.error("Could not find 'userId' column in users_df. Cannot create the full user ID set.")
                # Decide if this is a fatal error for the class
            except Exception as e:
                logging.error(f"Error processing user IDs from users_df: {e}")

    def run(self) -> Dict[str, Any]:
        # Send pipeline start notification
        self._send_notification(
            "PostgreSQL Upload Pipeline Started",
            f"Beginning database upload pipeline: **{self.name}**",
            color=0x0099FF,  # Blue for start
            fields=[
                {"name": "Target Database", "value": f"`{self.db_config.get('database', 'unknown')}`", "inline": True},
                {"name": "Host", "value": f"`{self.db_config.get('host', 'localhost')}`", "inline": True},
                {"name": "Data Sources", "value": f"{len(self.data_paths)} files configured", "inline": True},
                {"name": "Transaction Mode", "value": "Enabled (rollback on error)", "inline": True}
            ]
        )
        
        outputs = {}
        try:
            self._connect_db()
            self._load_data()

            # Call individual processing methods here in order
            # self._process_users()
            # self._process_authors()
            # ... and so on for all data types

            # self.conn.commit() # Commit after all successful operations
            logging.info("Data processing and insertion pipeline completed successfully.")
            outputs["status"] = "success"
            
            # Send success notification
            total_records = 0
            if hasattr(self, 'books_df') and self.books_df is not None:
                total_records += len(self.books_df)
            if hasattr(self, 'interactions_df') and self.interactions_df is not None:
                total_records += len(self.interactions_df)
            if hasattr(self, 'reviews_df') and self.reviews_df is not None:
                total_records += len(self.reviews_df)
            if hasattr(self, 'authors_df') and self.authors_df is not None:
                total_records += len(self.authors_df)
            if hasattr(self, 'users_df') and self.users_df is not None:
                total_records += len(self.users_df)
                
            self._send_notification(
                "PostgreSQL Upload Pipeline Complete! 🎉",
                f"Successfully completed database upload pipeline: **{self.name}**",
                color=0x00FF00,  # Green for success
                fields=[
                    {"name": "Database Connection", "value": "✅ Successful", "inline": True},
                    {"name": "Data Loading", "value": "✅ Successful", "inline": True},
                    {"name": "Processing Status", "value": "✅ Successful", "inline": True},
                    {"name": "Total Records Processed", "value": f"{total_records:,}", "inline": True},
                    {"name": "Transaction Status", "value": "✅ Committed", "inline": True},
                    {"name": "Database", "value": f"`{self.db_config.get('database', 'unknown')}`", "inline": True}
                ]
            )
        except psycopg2.DatabaseError as db_err:
            error_msg = f"Database error during processing: {db_err}"
            logging.error(error_msg)
            if self.conn: self.conn.rollback()
            outputs["status"] = "db_error"
            
            self._send_notification(
                "PostgreSQL Upload Failed - Database Error",
                error_msg,
                error=True,
                fields=[
                    {"name": "Error Type", "value": "Database Error", "inline": True},
                    {"name": "Transaction", "value": "🔄 Rolled back", "inline": True},
                    {"name": "Database", "value": f"`{self.db_config.get('database', 'unknown')}`", "inline": True}
                ]
            )
        except ValueError as val_err:
             error_msg = f"Configuration or Value error: {val_err}"
             logging.error(error_msg, exc_info=True)
             if self.conn: self.conn.rollback()
             outputs["status"] = "value_error"
             
             self._send_notification(
                 "PostgreSQL Upload Failed - Configuration Error",
                 error_msg,
                 error=True,
                 fields=[
                     {"name": "Error Type", "value": "Configuration/Value Error", "inline": True},
                     {"name": "Transaction", "value": "🔄 Rolled back", "inline": True}
                 ]
             )
        except Exception as e:
            error_msg = f"An unexpected error occurred in the pipeline: {e}"
            logging.error(error_msg, exc_info=True)
            if self.conn: self.conn.rollback()
            outputs["status"] = "error"
            
            self._send_notification(
                "PostgreSQL Upload Pipeline Failed",
                error_msg,
                error=True,
                fields=[
                    {"name": "Error Type", "value": "Unexpected Error", "inline": True},
                    {"name": "Transaction", "value": "🔄 Rolled back", "inline": True}
                ]
            )
        finally:
            self._close_db()
            
            # Send connection closure notification
            self._send_notification(
                "Database Connection Closed",
                "PostgreSQL connection properly closed",
                color=0x808080,  # Gray for cleanup
                fields=[
                    {"name": "Connection Status", "value": "🔌 Disconnected", "inline": True},
                    {"name": "Pipeline Status", "value": outputs.get("status", "unknown"), "inline": True}
                ]
            )
        self.output_data = outputs
        return outputs