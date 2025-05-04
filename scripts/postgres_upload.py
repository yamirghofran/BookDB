import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
import names
import pandas as pd
import dask.dataframe as dd
import random
import psycopg2
import psycopg2.extras as extras
import uuid
from faker import Faker
import re
import logging
from datetime import datetime
import pytz # Required for handling timezones robustly
import ast # For safely evaluating string representations of lists

POSTGRESQL_URL = os.getenv("POSTGRESQL_URL")
if POSTGRESQL_URL is None:
    raise ValueError("POSTGRESQL_URL environment variable not set")
engine = create_engine(POSTGRESQL_URL)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load data from datasets
books_df = pd.read_parquet('data/reduced_books.parquet')
interactions_df = pd.read_parquet('data/reduced_interactions.parquet')
reviews_df = pd.read_parquet('data/reduced_reviews.parquet')
authors_df = pd.read_parquet('data/new_authors.parquet')
users_df = pd.read_csv('data/user_id_map.csv')

dataframes = {
    'Books': books_df,
    'Interactions': interactions_df,
    'Reviews': reviews_df,
    'Authors': authors_df,
    'Users': users_df
}



# Print the heads of the dataframes
# for name, df in dataframes.items():
#     print(f"=========Head and types of {name} dataframe=========")
#     print(df.head())
#     print("\nTypes")
#     print(df.dtypes)
#     print("\n")

# --- Configuration ---
# Database connection parameters (replace with your actual credentials)
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT") # e.g., 5432
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Faker for generating fake user data
fake = Faker()

# Function to extract genres from popular_shelves
def extract_genres(popular_shelves):
    """
    Extracts potential genres from a list of popular shelves dictionaries,
    adding only the base genre keyword found.

    Args:
        popular_shelves: A list of dictionaries, where each dictionary has
                         'count' and 'name' keys. Can also be None or empty.

    Returns:
        A list of unique lowercase base genre names found, or an empty list on error.
    """
    try:
        # MODIFIED: Check if it's a list instead of np.ndarray
        if not isinstance(popular_shelves, list) or len(popular_shelves) == 0:
            return []

        # Use a set to store unique base genres found
        found_genres = set()

        # Using your comprehensive keyword lists
        genre_keywords = [
            'action', 'adventure', 'comedy', 'crime', 'mystery', 'textbook', 'children', 'mathematics', 'fantasy',
            'historical', 'horror', 'romance', 'satire', 'science fiction',
            'scifi', 'speculative fiction', 'thriller', 'western', 'paranormal',
            'dystopian', 'urban fantasy', 'contemporary', 'young adult', 'ya',
            'middle grade', 'children\'s', 'literary fiction', 'magic realism',
            'historical fiction', 'gothic', 'suspense', 'biography', 'memoir',
            'nonfiction', 'poetry', 'drama', 'historical romance',
            'fantasy romance', 'romantic suspense', 'science fiction romance',
            'contemporary romance', 'paranormal romance', 'epic fantasy',
            'dark fantasy', 'sword and sorcery', 'steampunk', 'cyberpunk',
            'apocalyptic', 'post-apocalyptic', 'alternate history',
            'superhero', 'mythology', 'fairy tales', 'folklore', 'war',
            'military fiction', 'spy fiction', 'political fiction', 'social science fiction',
            'techno-thriller', 'medical thriller', 'legal thriller',
            'psychological thriller', 'cozy mystery', 'hardboiled', 'noir',
            'coming-of-age', 'lgbtq+', 'christian fiction', 'religious fiction',
            'humor', 'travel', 'food', 'cooking', 'health', 'self-help',
            'business', 'finance', 'history', 'science', 'technology', 'nature',
            'art', 'music', 'philosophy', 'education', 'true crime', 'spiritual',
            'anthology', 'short stories', 'plays', 'screenplays', 'graphic novel',
            'comics', 'manga', 'erotica', 'new adult', 'chick lit', 'womens fiction',
            'sports fiction', 'family saga', ' Regency romance', 'literature'
        ]
        genre_keywords.sort(key=len, reverse=True)

        ignore_keywords = ['to-read', 'owned', 'hardcover', 'shelfari-favorites', 'series', 'might-read',
                           'dnf-d', 'hambly-barbara', 'strong-females', 'first-in-series',
                           'no-thanks-series-collections-boxes', 'entertaining-but-limited',
                           'kate-own', 'e-book', 'compliation', 'my-books',
                           'books-i-own-but-have-not-read', 'everything-owned', 'books-to-find',
                           'i-own-it', 'favorite', 'not-read', 'read-some-day', 'library',
                           'audiobooks', 'status-borrowed', 'owned-books',
                           'spec-fic-awd-locus-nom', '01', 'hardbacks', 'paper', 'german',
                           'hardback', 'physical-scifi-fantasy', 'childhood-favorites',
                           'bundle-same-author', 'aa-sifi-fantasy', 'ready-to-read',
                           'bought-on-flee-markets', 'fantasy-general', 'hardcopy', 'box-2',
                           'unfinished', 'magic', 'duplicates', 'favorites', 'books-i-own',
                           'fantasy-classic', 'own-hard-copy', 'fantasy-read',
                           'book-club-edition', 'sci-fi-or-fantasy', 'fiction-fantasy',
                           'fiction-literature-poetry', 'paused-hiatus', 'status—borrowed',
                           'recs-fantasy', 'fantasy-scifi', 'omnibus', 'speculative',
                           'sf--fantasy', 'in-my-home-library', 'fant-myth-para-vamps',
                           'read-in-my-20s'] # Your ignore list

        for shelf in popular_shelves:
            # Robust check for dict with 'name' key
            if not isinstance(shelf, dict) or 'name' not in shelf or not isinstance(shelf['name'], str):
                continue

            shelf_name = shelf['name'].lower().strip() # Normalize shelf name

            # Skip if shelf name contains any ignore keywords
            # Use word boundaries for more precise ignore matching if needed later
            # e.g., if r'\b' + re.escape(ignore) + r'\b' in shelf_name:
            if any(ignore in shelf_name for ignore in ignore_keywords):
                continue

            # Check if any genre keyword is present in the shelf name
            for keyword in genre_keywords:
                # Simple substring check is kept as per your function
                # Consider word boundary checks ( re.search(r'\b' + re.escape(keyword) + r'\b', shelf_name) )
                # if more precision is needed to avoid 'art' matching 'heart'.
                if keyword in shelf_name:
                    found_genres.add(keyword) # Add the base keyword (lowercase)
                    # Optional: break here if only first/longest match per shelf needed
                    # break

        # Return sorted list of lowercase genres
        return sorted(list(found_genres))
    except Exception as e:
        # Using logging from the main script setup
        logging.error(f"Error in extract_genres function processing data: {popular_shelves[:5]}... - Error: {e}", exc_info=True)
        return [] # Return empty list on error


# --- Helper Functions ---
def safe_to_int(value, default=None):
    """Safely convert value to integer, returning default if conversion fails."""
    try:
        return int(value) if pd.notna(value) else default
    except (ValueError, TypeError):
        return default

def safe_to_float(value, default=None):
    """Safely convert value to float, returning default if conversion fails."""
    try:
        return float(value) if pd.notna(value) else default
    except (ValueError, TypeError):
        return default

def parse_goodreads_date(date_str, default=None):
    """
    Parses the specific Goodreads date format string into a timezone-aware datetime object.
    Format: 'Mon Mar 20 23:58:16 -0700 2017'
    """
    if pd.isna(date_str) or not date_str:
        return default
    try:
        # Extract timezone offset correctly
        dt_part = date_str[:-10].strip()
        tz_part = date_str[-10:].strip()
        offset_str = tz_part[:5] # e.g., '-0700'
        year_str = tz_part[5:].strip()

        # Reconstruct string for strptime, handling the offset manually
        naive_dt = datetime.strptime(f"{dt_part} {year_str}", '%a %b %d %H:%M:%S %Y')

        # Create timezone object
        offset_hours = int(offset_str[:3])
        offset_minutes = int(offset_str[0] + offset_str[3:]) # Keep sign for minutes
        tz = pytz.FixedOffset((offset_hours * 60) + offset_minutes)

        # Make datetime timezone-aware
        aware_dt = tz.localize(naive_dt)
        return aware_dt
    except (ValueError, TypeError, IndexError) as e:
        logging.warning(f"Could not parse date string '{date_str}': {e}")
        return default

def parse_list_string(list_str, default=None):
    """Safely parses a string representation of a list using ast.literal_eval."""
    if pd.isna(list_str) or not isinstance(list_str, str) or not list_str.strip():
         return default if default is not None else []
    try:
        # Using ast.literal_eval is safer than eval()
        result = ast.literal_eval(list_str)
        # Ensure the result is actually a list
        return result if isinstance(result, list) else (default if default is not None else [])
    except (ValueError, SyntaxError, TypeError) as e:
         logging.warning(f"Could not parse list string '{str(list_str)[:100]}...': {e}")
         return default if default is not None else []
# --- Main Data Processing and Insertion Function ---
def process_and_insert_data(conn):
    """Processes dataframes and inserts data into the PostgreSQL database."""

    cursor = conn.cursor()
    logging.info("Starting data processing and insertion...")

    # --- 1. Process and Insert Users ---
    logging.info("Processing Users...")
    user_mapping = {} # goodreads_user_id -> new_user_uuid
    users_to_insert = []
    processed_goodreads_ids = set() # Keep track to avoid duplicates if any

    # Ensure 'user_id' is string type for consistent processing
    users_df['user_id'] = users_df['user_id'].astype(str)

    for _, row in users_df.iterrows():
        goodreads_user_id = row['user_id']
        if goodreads_user_id in processed_goodreads_ids or pd.isna(goodreads_user_id):
            continue

        new_user_uuid = uuid.uuid4()
        user_mapping[goodreads_user_id] = new_user_uuid
        users_to_insert.append((
            new_user_uuid,
            fake.name(), # Generate fake name
            fake.unique.email() # Generate unique fake email
        ))
        processed_goodreads_ids.add(goodreads_user_id)

    logging.info(f"Inserting {len(users_to_insert)} users...")
    try:
        extras.execute_values(
            cursor,
            "INSERT INTO Users (id, name, email) VALUES %s ON CONFLICT (email) DO NOTHING",
            users_to_insert,
            template="(%s, %s, %s)"
        )
    except Exception as e:
        logging.error(f"Error inserting users: {e}")
        conn.rollback() # Rollback on error
        return

    # --- 2. Process and Insert Authors ---
    logging.info("Processing Authors...")
    author_mapping = {} # goodreads_author_id -> new_author_uuid
    authors_to_insert = []
    processed_goodreads_ids = set()

    # Ensure 'author_id' is string type
    authors_df['author_id'] = authors_df['author_id'].astype(str)

    for _, row in authors_df.iterrows():
        goodreads_author_id = row['author_id']
        if goodreads_author_id in processed_goodreads_ids or pd.isna(goodreads_author_id):
            continue

        new_author_uuid = uuid.uuid4()
        author_mapping[goodreads_author_id] = new_author_uuid
        authors_to_insert.append((
            new_author_uuid,
            row['name'],
            safe_to_float(row['average_rating']),
            safe_to_int(row['ratings_count'])
        ))
        processed_goodreads_ids.add(goodreads_author_id)

    logging.info(f"Inserting {len(authors_to_insert)} authors...")
    try:
        extras.execute_values(
            cursor,
            "INSERT INTO Authors (id, name, average_rating, ratings_count) VALUES %s",
            authors_to_insert,
            template="(%s, %s, %s, %s)"
        )
    except Exception as e:
        logging.error(f"Error inserting authors: {e}")
        conn.rollback()
        return

    # --- 3. Process Genres and Books ---
    logging.info("Processing Books and Genres...")
    book_mapping = {} # goodreads_book_id -> new_book_uuid
    books_to_insert = []
    book_author_relations = []
    # book_genre_relations = [] # We generate this later after inserting genres
    similar_book_relations = set()
    all_genres = set() # Collect all unique genres (lowercase)
    processed_goodreads_ids = set()

    # Pre-process columns (includes parsing lists needed later)
    books_df['goodreads_id'] = books_df['book_id'].astype(int)
    books_df['publication_year'] = books_df['publication_year'].apply(lambda x: safe_to_int(x))
    books_df['average_rating'] = books_df['average_rating'].apply(lambda x: safe_to_float(x))
    books_df['ratings_count'] = books_df['ratings_count'].apply(lambda x: safe_to_int(x))
    # Parse list strings safely
    books_df['authors_list'] = books_df['authors'].apply(parse_list_string)
    books_df['similar_books_list'] = books_df['similar_books'].apply(parse_list_string)
    # Parse popular_shelves into a list of dicts for the genre function
    books_df['popular_shelves_list'] = books_df['popular_shelves'].apply(parse_list_string) # Use the helper

    for _, row in books_df.iterrows():
        goodreads_book_id = row['goodreads_id']
        if goodreads_book_id in processed_goodreads_ids or pd.isna(goodreads_book_id):
            continue

        new_book_uuid = uuid.uuid4()
        book_mapping[goodreads_book_id] = new_book_uuid

        books_to_insert.append((
            new_book_uuid,
            goodreads_book_id,
            row['url'],
            row['title_without_series'],
            row['description'],
            row['publication_year'],
            row['image_url'],
            row['average_rating'],
            row['ratings_count']
        ))
        processed_goodreads_ids.add(goodreads_book_id)

        # Map Authors
        for goodreads_author_id in row['authors_list']:
             if isinstance(goodreads_author_id, str) and goodreads_author_id in author_mapping:
                book_author_relations.append((new_book_uuid, author_mapping[goodreads_author_id]))

        # Extract Genres using your function with the parsed list
        # Pass the pre-parsed list from 'popular_shelves_list' column
        genres = extract_genres(row['popular_shelves_list']) # <--- USE YOUR FUNCTION HERE
        all_genres.update(genres) # Collect all unique lowercase genres

        # Prepare Similar Books (mapping done later)
        for similar_goodreads_id in row['similar_books_list']:
            # Ensure it's an int before adding
            sim_id = None
            if isinstance(similar_goodreads_id, int):
                sim_id = similar_goodreads_id
            elif isinstance(similar_goodreads_id, str) and similar_goodreads_id.isdigit():
                 sim_id = int(similar_goodreads_id)

            if sim_id is not None and goodreads_book_id != sim_id:
                pair = tuple(sorted((goodreads_book_id, sim_id)))
                similar_book_relations.add(pair)


    # --- 4. Insert Genres ---
    logging.info("Inserting Genres...")
    # Ensure genres are lowercase and prepare for insertion
    genres_to_insert = [(genre,) for genre in all_genres if genre] # Already lowercase
    genre_mapping = {} # genre_name (lowercase) -> genre_id

    if genres_to_insert:
        try:
            cursor.executemany(
                # Ensure your Genres table name column handles case sensitivity appropriately
                # If it's case sensitive, you might need lower() in queries or ensure consistency.
                # Assuming standard case-insensitive comparison or lowercase storage.
                "INSERT INTO Genres (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                genres_to_insert
            )
            # Fetch all genres (lowercase name) to get their IDs
            cursor.execute("SELECT name, id FROM Genres WHERE name = ANY(%s)", (list(all_genres),))
            genre_mapping = {name: genre_id for name, genre_id in cursor.fetchall()}
            logging.info(f"Upserted/found {len(genre_mapping)} unique genres.")
        except Exception as e:
            logging.error(f"Error inserting/fetching genres: {e}")
            conn.rollback()
            return

    # --- 5. Insert Books ---
    logging.info(f"Inserting {len(books_to_insert)} books...")
    try:
        extras.execute_values(
            cursor,
            """INSERT INTO Books (id, goodreads_id, goodreads_url, title, description,
                                publication_year, cover_image_url, average_rating, ratings_count)
               VALUES %s""",
            books_to_insert,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
    except Exception as e:
        logging.error(f"Error inserting books: {e}")
        conn.rollback()
        return

    # --- 6. Insert Book-Author Relations ---
    logging.info(f"Inserting {len(book_author_relations)} book-author relations...")
    if book_author_relations:
      try:
          extras.execute_values(
              cursor,
              "INSERT INTO BookAuthors (book_id, author_id) VALUES %s ON CONFLICT DO NOTHING",
              book_author_relations,
              template="(%s, %s)"
          )
      except Exception as e:
          logging.error(f"Error inserting book-author relations: {e}")
          conn.rollback()
          return

    # --- 7. Insert Book-Genre Relations ---
    logging.info("Mapping and Inserting Book-Genre relations...")
    book_genre_relations_to_insert = []
    # Re-iterate books df to map genres using the populated genre_mapping
    for _, row in books_df.iterrows():
        goodreads_book_id = row['goodreads_id']
        if goodreads_book_id in book_mapping:
            book_uuid = book_mapping[goodreads_book_id]
            # Extract genres again using the parsed list
            genres = extract_genres(row['popular_shelves_list'])
            for genre_name in genres: # genre_name is lowercase here
                if genre_name in genre_mapping: # Check against lowercase keys in mapping
                    book_genre_relations_to_insert.append((book_uuid, genre_mapping[genre_name]))
                else:
                     logging.warning(f"Genre '{genre_name}' extracted for book {goodreads_book_id} but not found in genre mapping. Skipping relation.")


    if book_genre_relations_to_insert:
      logging.info(f"Inserting {len(book_genre_relations_to_insert)} book-genre relations...")
      try:
          extras.execute_values(
              cursor,
              "INSERT INTO BookGenres (book_id, genre_id) VALUES %s ON CONFLICT DO NOTHING",
              book_genre_relations_to_insert,
              template="(%s, %s)"
          )
      except Exception as e:
          logging.error(f"Error inserting book-genre relations: {e}")
          conn.rollback()
          return

    # --- 8. Insert Similar Book Relations ---
    logging.info("Mapping and Inserting Similar Book relations...")
    similar_books_to_insert = []
    for gr_id1, gr_id2 in similar_book_relations:
        if gr_id1 in book_mapping and gr_id2 in book_mapping:
            uuid1 = book_mapping[gr_id1]
            uuid2 = book_mapping[gr_id2]
            # Ensure ordered pair using UUIDs for the final check constraint
            # Although sorting by Goodreads ID already helps, UUIDs might not sort the same way
            # The DB constraint `check_ordered_pair` handles this by checking book_id_1 < book_id_2
            # We should insert them in the correct order based on UUID comparison.
            if uuid1 < uuid2:
                 similar_books_to_insert.append((uuid1, uuid2))
            else:
                 similar_books_to_insert.append((uuid2, uuid1)) # Insert smaller UUID first

    if similar_books_to_insert:
        logging.info(f"Inserting {len(similar_books_to_insert)} similar book relations...")
        try:
            extras.execute_values(
                cursor,
                "INSERT INTO SimilarBooks (book_id_1, book_id_2) VALUES %s ON CONFLICT DO NOTHING",
                similar_books_to_insert,
                template="(%s, %s)"
            )
        except Exception as e:
            logging.error(f"Error inserting similar book relations: {e}")
            conn.rollback()
            return


    # --- 9. Process and Insert Reviews ---
    logging.info("Processing and Inserting Reviews...")
    reviews_to_insert = []
    processed_review_ids = set()

    # Ensure correct types
    reviews_df['user_id'] = reviews_df['user_id'].astype(str)
    reviews_df['book_id'] = reviews_df['book_id'].astype(int)
    reviews_df['rating'] = reviews_df['rating'].apply(lambda x: safe_to_int(x, default=0)) # Use 0 if invalid? Schema allows NULL. Let's use None.
    reviews_df['rating'] = reviews_df['rating'].apply(lambda x: x if x in range(1, 6) else None) # Enforce 1-5 range, else NULL
    reviews_df['review_text'] = reviews_df['review_text'].astype(str).fillna('') # Ensure string, handle NaN

    for _, row in reviews_df.iterrows():
        # Check if review_id exists and is unique (assuming review_id is unique identifier in source)
        goodreads_review_id = row['review_id']
        if pd.isna(goodreads_review_id) or goodreads_review_id in processed_review_ids:
             continue

        goodreads_user_id = row['user_id']
        goodreads_book_id = row['book_id']

        # Only insert if user and book were successfully processed and mapped
        if goodreads_user_id in user_mapping and goodreads_book_id in book_mapping:
            new_review_uuid = uuid.uuid4()
            reviews_to_insert.append((
                new_review_uuid,
                row['review_text'],
                row['rating'], # Already cleaned to be 1-5 or None
                book_mapping[goodreads_book_id],
                user_mapping[goodreads_user_id]
                # created_at/updated_at handled by DB defaults
            ))
            processed_review_ids.add(goodreads_review_id)

    if reviews_to_insert:
      logging.info(f"Inserting {len(reviews_to_insert)} reviews...")
      try:
          extras.execute_values(
              cursor,
              "INSERT INTO Reviews (id, text, rating, book_id, user_id) VALUES %s",
              reviews_to_insert,
              template="(%s, %s, %s, %s, %s)"
          )
      except Exception as e:
          logging.error(f"Error inserting reviews: {e}")
          conn.rollback()
          return

    # --- 10. Process and Insert User Library Entries ---
    logging.info("Processing and Inserting User Library entries...")
    library_entries_to_insert = []
    processed_library_pairs = set() # To avoid duplicate user-book pairs

    # Ensure correct types
    interactions_df['user_id'] = interactions_df['user_id'].astype(str)
    interactions_df['book_id'] = interactions_df['book_id'].astype(int)
    # interactions_df['date_added_dt'] = interactions_df['date_added'].apply(parse_goodreads_date) # Parse dates

    for _, row in interactions_df.iterrows():
        goodreads_user_id = row['user_id']
        goodreads_book_id = row['book_id']
        user_book_pair = (goodreads_user_id, goodreads_book_id)

        if user_book_pair in processed_library_pairs:
            continue

        # Only insert if user and book exist in our mappings
        if goodreads_user_id in user_mapping and goodreads_book_id in book_mapping:
            added_at_dt = parse_goodreads_date(row['date_added'])
            library_entries_to_insert.append((
                user_mapping[goodreads_user_id],
                book_mapping[goodreads_book_id],
                added_at_dt # Use parsed date or None if parsing failed
            ))
            processed_library_pairs.add(user_book_pair)

    if library_entries_to_insert:
      logging.info(f"Inserting {len(library_entries_to_insert)} user library entries...")
      try:
          # Use ON CONFLICT to avoid errors if a user-book pair somehow appears twice
          # Update added_at if the new date is earlier (or just keep the first one)
          extras.execute_values(
              cursor,
              """INSERT INTO UserLibrary (user_id, book_id, added_at) VALUES %s
                 ON CONFLICT (user_id, book_id) DO NOTHING""", # Or DO UPDATE SET added_at = EXCLUDED.added_at
              library_entries_to_insert,
              template="(%s, %s, %s)"
          )
      except Exception as e:
          logging.error(f"Error inserting user library entries: {e}")
          conn.rollback()
          return

    # --- Commit Transaction ---
    conn.commit()
    logging.info("Successfully inserted all data and committed transaction.")

# --- Main Execution Block ---
if __name__ == "__main__":
    conn = None
    try:
        logging.info("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False # Ensure operations are transactional
        psycopg2.extras.register_uuid() # Register UUID adapter for psycopg2

        # Optional: Enable UUID extension if not already enabled
        # with conn.cursor() as cur:
        #     cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
        # conn.commit() # Commit extension creation separately
        process_and_insert_data(conn)

    except psycopg2.DatabaseError as db_err:
        logging.error(f"Database error: {db_err}")
        if conn:
            conn.rollback() # Rollback any changes if error occurs during connection or processing
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")