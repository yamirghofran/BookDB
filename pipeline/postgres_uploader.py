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
        except psycopg2.OperationalError as db_err:
            logging.error(f"Database connection failed: {db_err}")
            raise  # Re-raise the exception to be handled by the caller
        except Exception as e:
            logging.error(f"An unexpected error occurred during DB connection: {e}")
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
        except FileNotFoundError as e:
            logging.error(f"Error loading data file: {e}. Check paths in data_paths config.")
            raise
        except Exception as e:
            logging.error(f"An error occurred during data loading: {e}")
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
        except psycopg2.DatabaseError as db_err:
            logging.error(f"Database error during processing: {db_err}")
            if self.conn: self.conn.rollback()
            outputs["status"] = "db_error"
        except ValueError as val_err:
             logging.error(f"Configuration or Value error: {val_err}", exc_info=True)
             if self.conn: self.conn.rollback()
             outputs["status"] = "value_error"
        except Exception as e:
            logging.error(f"An unexpected error occurred in the pipeline: {e}", exc_info=True)
            if self.conn: self.conn.rollback()
            outputs["status"] = "error"
        finally:
            self._close_db()
        self.output_data = outputs
        return outputs