import os
from dotenv import load_dotenv
load_dotenv()
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

# --- Environment/Config Loading ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT") # e.g., 5432
}

# --- Data Loading ---
try:
    books_df = pd.read_parquet('data/reduced_books.parquet')
    interactions_df = pd.read_parquet('data/reduced_interactions.parquet')
    reviews_df = pd.read_parquet('data/reduced_reviews.parquet')
    authors_df = pd.read_parquet('data/new_authors.parquet')
    users_df = pd.read_csv('data/ncf_user_id_map_reduced.csv') # Using the specified CSV
    item_id_map_df = pd.read_csv('data/ncf_item_id_map_reduced.csv')  # Load book ncf_id mapping
except FileNotFoundError as e:
    logging.error(f"Error loading data file: {e}. Make sure all files are in the 'data/' directory.")
    exit(1)
except Exception as e:
    logging.error(f"An error occurred during data loading: {e}")
    exit(1)

# --- Logging & Faker Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
fake = Faker()

# --- Genre Extraction Function ---
# (Implementation unchanged)
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
                if keyword in shelf_name: found_genres.add(keyword) # break # Optional
        return sorted(list(found_genres))
    except Exception as e:
        logging.error(f"Error in extract_genres function processing data: {popular_shelves[:5]}... - Error: {e}", exc_info=True); return []

# --- Helper Functions ---
# (safe_to_int, safe_to_float, parse_goodreads_date, parse_list_string - unchanged)
def safe_to_int(value, default=None):
    try: return int(value) if pd.notna(value) else default
    except (ValueError, TypeError): return default

def safe_to_float(value, default=None):
    try: return float(value) if pd.notna(value) else default
    except (ValueError, TypeError): return default

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

def parse_list_string(list_str, default=None):
    default_value = default if default is not None else []

    # 1. Handle actual strings
    if isinstance(list_str, str):
        if not list_str.strip():
            return default_value # Return default for empty strings
        try:
            # Attempt to parse the non-empty string
            result = ast.literal_eval(list_str)
            # Ensure the parsed result is actually a list
            return result if isinstance(result, list) else default_value
        except (ValueError, SyntaxError, TypeError) as e:
            logging.warning(f"Could not parse list string '{str(list_str)[:100]}...': {e}")
            return default_value
    # 2. Handle inputs that are already lists
    elif isinstance(list_str, list):
         return list_str # It's already a list, return as is
    # 3. Handle NumPy arrays
    elif isinstance(list_str, np.ndarray):
         return list_str.tolist() # Convert numpy array to Python list
    # 4. Handle scalar NaN/None values (check after array types)
    elif pd.isna(list_str):
        return default_value
    # 5. Handle other unexpected types
    else:
        logging.warning(f"Unexpected type for parse_list_string: {type(list_str)}. Value: {str(list_str)[:100]}...")
        return default_value

def is_valid_uuid(uuid_to_test, version=4):
    """Check if uuid_to_test is a valid UUID string."""
    if isinstance(uuid_to_test, uuid.UUID): return True
    try: uuid.UUID(str(uuid_to_test), version=version); return True
    except (ValueError, TypeError, AttributeError): return False


# --- Main Data Processing and Insertion Function ---
def process_and_insert_data(conn, users_df, authors_df, books_df, reviews_df, interactions_df):
    """Processes dataframes and inserts data into the PostgreSQL database."""
    cursor = conn.cursor()
    logging.info("Starting data processing and insertion...")

    # --- 1. Process and Insert Users ---
    logging.info("Processing Users...")
    inserted_user_ids = set()
    users_to_insert = []
    processed_original_user_ids = set()

    # --- Populate the set of all expected user IDs --- # <--- NEW
    try:
        all_input_user_ids = set(users_df['userId'].astype(str).unique()) # Assuming 'user_id' is the column with UUIDs
        logging.info(f"Collected {len(all_input_user_ids)} unique user IDs from the input users_df.")
    except KeyError:
        logging.error("Could not find 'user_id' column in users_df. Cannot create the full user ID set.")
        return # Or handle error appropriately
    except Exception as e:
        logging.error(f"Error processing user IDs from users_df: {e}")
        return
    # --- End NEW section ---

    users_df['userId'] = users_df['userId'].astype(str)
    reviews_df['user_id'] = reviews_df['user_id'].astype(str)
    interactions_df['user_id'] = interactions_df['user_id'].astype(str)

    # Generate a fixed fake domain name once for consistency in this run
    # Or use fake.domain_name() inside the loop if variation is desired
    fake_domain = fake.domain_name()

    for _, row in users_df.iterrows():
        original_user_id_str = row['userId']
        ncf_id = row['ncf_userId']

        if original_user_id_str in processed_original_user_ids: continue
        processed_original_user_ids.add(original_user_id_str)

        if not is_valid_uuid(original_user_id_str):
            logging.warning(f"Skipping user row: Invalid UUID format for userId '{original_user_id_str}'.")
            continue

        # --- MODIFIED Email Generation ---
        # Create a unique email based on the user's UUID
        user_email = f"{original_user_id_str}@{fake_domain}"
        # We no longer need fake.unique.email()
        #--------------------------------

        users_to_insert.append((
            original_user_id_str,
            ncf_id,
            fake.name(),
            user_email  # Use the generated UUID-based email
        ))

    

    logging.info(f"Inserting {len(users_to_insert)} users...")
    if users_to_insert:
        try:
            # --- Check for Duplicates within the Batch and Pre-existing Emails ---
            all_emails_in_batch = [email for _, _, _, email in users_to_insert]
            unique_emails_in_batch_set = set(all_emails_in_batch)

            # Check 1: Duplicates within the current batch
            if len(all_emails_in_batch) != len(unique_emails_in_batch_set):
                num_duplicates = len(all_emails_in_batch) - len(unique_emails_in_batch_set)
                logging.warning(f"Detected {num_duplicates} duplicate email(s) within the current batch of users to insert. "
                                f"Only the first occurrence of each email will be inserted due to ON CONFLICT.")
                # Optional: Find and log example duplicates if needed for debugging
                # from collections import Counter
                # email_counts = Counter(all_emails_in_batch)
                # duplicate_examples = [email for email, count in email_counts.items() if count > 1][:5]
                # logging.warning(f"Example duplicate emails in batch: {duplicate_examples}")

            # Check 2: Conflicts with existing data in the DB
            logging.info(f"Checking {len(unique_emails_in_batch_set)} unique generated emails against existing DB emails...")
            # Only query the DB with the unique emails from the batch
            cursor.execute("SELECT email FROM Users WHERE email = ANY(%s)", (list(unique_emails_in_batch_set),))
            existing_conflicting_emails = {row[0] for row in cursor.fetchall()}

            if existing_conflicting_emails:
                logging.warning(f"Detected {len(existing_conflicting_emails)} email conflicts with existing data in the Users table. "
                                f"These users will be skipped due to ON CONFLICT. "
                                f"Example conflicting emails: {list(existing_conflicting_emails)[:5]}")
            elif len(unique_emails_in_batch_set) > 0: # Avoid logging if the batch was empty after filtering
                 logging.info("No email conflicts detected with existing data in the Users table.")
            # --- End Check ---

            user_insert_tuples = [(uid, ncf_id, name, email) for uid, ncf_id, name, email in users_to_insert]
            extras.execute_values(
                cursor,
                # Note: Even with UUID-based emails, ON CONFLICT is kept as a safety measure,
                # in case the script is run with the same users_df multiple times or if IDs repeat.
                "INSERT INTO Users (id, ncf_id, name, email) VALUES %s ON CONFLICT (email) DO NOTHING RETURNING id",
                user_insert_tuples,
                template="(%s, %s, %s, %s)"
            )
            inserted_ids_from_db = [item[0] for item in cursor.fetchall()]
            inserted_user_ids.update(inserted_ids_from_db)
            logging.info(f"Successfully processed {len(inserted_user_ids)} user IDs into DB.") # This count should now be closer to len(users_to_insert) if DB was empty

        except Exception as e:
            logging.error(f"Error inserting users: {e}"); conn.rollback(); return
    else:
        logging.warning("No valid users found in users_df to process.")


    # --- 2. Process and Insert Authors ---
    # (Implementation unchanged)
    logging.info("Processing Authors...")
    author_mapping = {}; authors_to_insert = []; processed_goodreads_author_ids = set()
    authors_df['author_id'] = authors_df['author_id'].astype(str)
    for _, row in authors_df.iterrows():
        goodreads_author_id = row['author_id']
        if goodreads_author_id in processed_goodreads_author_ids or pd.isna(goodreads_author_id): continue
        new_author_uuid = uuid.uuid4(); author_mapping[goodreads_author_id] = new_author_uuid
        authors_to_insert.append((new_author_uuid, row['name'], safe_to_float(row['average_rating']), safe_to_int(row['ratings_count'])))
        processed_goodreads_author_ids.add(goodreads_author_id)
    logging.info(f"Inserting {len(authors_to_insert)} authors...")
    if authors_to_insert:
        try: extras.execute_values(cursor, "INSERT INTO Authors (id, name, average_rating, ratings_count) VALUES %s", authors_to_insert, template="(%s, %s, %s, %s)")
        except Exception as e: logging.error(f"Error inserting authors: {e}"); conn.rollback(); return
    else: logging.warning("No authors found to insert.")


    # --- 3. Process Genres and Books ---
    logging.info("Processing Books and Genres...")
    book_mapping = {}
    books_to_insert = []
    book_author_relations = []
    similar_book_relations = set()
    all_genres = set()
    processed_goodreads_book_ids = set()

    # --- Book ncf_id Mapping ---
    book_ncf_id_map = dict(zip(item_id_map_df['itemId'], item_id_map_df['ncf_itemId']))

    # --- FIX: Added specific check for Pandas Ambiguity Error ---
    try:
        # Convert book_ids to integer type
        books_df['goodreads_id'] = books_df['book_id'].astype(int)
        reviews_df['book_id'] = reviews_df['book_id'].astype(int)
        interactions_df['book_id'] = interactions_df['book_id'].astype(int)
    except ValueError as ve:
        # Check if it's the specific ambiguity error
        if "ambiguous" in str(ve):
             logging.error(f"Pandas ambiguity error during book_id conversion: {ve}. "
                           "This usually means a 'book_id' column contains lists or arrays instead of simple values. "
                           "Please check the source data types in the 'book_id' columns of your Parquet files.")
        else: # Handle other potential ValueError during int conversion
             logging.error(f"ValueError during book_id conversion: {ve}. Check 'book_id' columns for non-numeric data.")
        conn.rollback()
        return
    except TypeError as te:
        logging.error(f"TypeError during book_id conversion: {te}. Check data types in 'book_id' columns.")
        conn.rollback()
        return
    # ----------------------------------------------------------

    # Continue with other pre-processing
    books_df['publication_year'] = books_df['publication_year'].apply(lambda x: safe_to_int(x))
    books_df['average_rating'] = books_df['average_rating'].apply(lambda x: safe_to_float(x))
    books_df['ratings_count'] = books_df['ratings_count'].apply(lambda x: safe_to_int(x))
    books_df['authors_list'] = books_df['authors'].apply(parse_list_string)
    books_df['similar_books_list'] = books_df['similar_books'].apply(parse_list_string)
    books_df['popular_shelves_list'] = books_df['popular_shelves'].apply(parse_list_string)

    # Main loop for processing books
    for _, row in books_df.iterrows():
        # Ensure goodreads_book_id is an integer or None
        try:
            # Use safe_to_int for robustness against potential non-numeric values
            goodreads_book_id = safe_to_int(row['goodreads_id'])
        except Exception as e: # Catch any unexpected error during conversion
             logging.warning(f"Skipping book row due to error converting goodreads_id '{row['goodreads_id']}': {e}")
             continue # Skip this book row

        if goodreads_book_id is None or goodreads_book_id in processed_goodreads_book_ids:
            continue

        new_book_uuid = uuid.uuid4()
        book_mapping[goodreads_book_id] = new_book_uuid

        # Explicitly prepare values with correct types or None
        pub_year = safe_to_int(row['publication_year'])
        avg_rating = safe_to_float(row['average_rating'])
        ratings_cnt = safe_to_int(row['ratings_count']) # Ensure this is int or None

        # --- Get ncf_id for this book ---
        ncf_id_raw = book_ncf_id_map.get(goodreads_book_id)
        ncf_id = safe_to_int(ncf_id_raw) if ncf_id_raw is not None else None

        books_to_insert.append((
            new_book_uuid,
            ncf_id,
            goodreads_book_id, # Already confirmed as int or None
            row['url'],
            row['title_without_series'],
            row['description'],
            pub_year,         # int or None
            row['image_url'],
            avg_rating,       # float or None
            ratings_cnt       # int or None
        ))
        processed_goodreads_book_ids.add(goodreads_book_id)

        # Process authors, genres, similar books (ensure they use the validated goodreads_book_id)
        for author_data in row['authors_list']:
             if isinstance(author_data, dict) and 'author_id' in author_data:
                 goodreads_author_id = str(author_data['author_id']) # Ensure string key
                 if goodreads_author_id in author_mapping:
                     book_author_relations.append((new_book_uuid, author_mapping[goodreads_author_id]))
             # Add handling if authors_list contains simple IDs directly (adjust as needed)
             elif isinstance(author_data, (str, int)):
                goodreads_author_id = str(author_data)
                if goodreads_author_id in author_mapping:
                   book_author_relations.append((new_book_uuid, author_mapping[goodreads_author_id]))


        genres = extract_genres(row['popular_shelves_list'])
        all_genres.update(genres)

        for similar_goodreads_id in row['similar_books_list']:
            sim_id = safe_to_int(similar_goodreads_id) # Use safe_to_int here too
            if sim_id is not None and goodreads_book_id != sim_id:
                pair = tuple(sorted((goodreads_book_id, sim_id)))
                similar_book_relations.add(pair)

    # --- 4. Insert Genres ---
    # (Implementation unchanged)
    logging.info("Inserting Genres...")
    genres_to_insert = [(genre,) for genre in all_genres if genre]; genre_mapping = {}
    if genres_to_insert:
        try:
            cursor.executemany("INSERT INTO Genres (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", genres_to_insert)
            cursor.execute("SELECT name, id FROM Genres WHERE name = ANY(%s)", (list(all_genres),)); genre_mapping = {name: genre_id for name, genre_id in cursor.fetchall()}
            logging.info(f"Upserted/found {len(genre_mapping)} unique genres.")
        except Exception as e: logging.error(f"Error inserting/fetching genres: {e}"); conn.rollback(); return
    else: logging.warning("No genres found to insert.")

    # --- 5. Insert Books ---
    # (Implementation unchanged)
    logging.info(f"Inserting {len(books_to_insert)} books...")
    if books_to_insert:
        try:
            if books_to_insert:
                print("Sample book tuple:", books_to_insert[0])
                print("Tuple length:", len(books_to_insert[0]))
                print("Expecting 10 columns for Books insert")
            # --- Add Debugging: Check min/max values before insert ---
            goodreads_ids = [b[2] for b in books_to_insert if len(b) > 2 and b[2] is not None] # Index 2 is goodreads_id
            ratings_counts = [b[9] for b in books_to_insert if len(b) > 9 and b[9] is not None] # Index 9 is ratings_count
            pub_years = [b[6] for b in books_to_insert if len(b) > 6 and b[6] is not None] # Index 6 is publication_year

            if books_to_insert:
                for i, book_tuple in enumerate(books_to_insert[:5]):
                    print(f"Book tuple {i}: {book_tuple} (length: {len(book_tuple)})")

            # Defensive: check all tuples are length 10
            bad_tuples = [i for i, t in enumerate(books_to_insert) if len(t) != 10]
            if bad_tuples:
                print(f"ERROR: Found {len(bad_tuples)} tuples with wrong length. Example indices: {bad_tuples[:5]}")
                for idx in bad_tuples[:5]:
                    print(f"Bad tuple at index {idx}: {books_to_insert[idx]} (length: {len(books_to_insert[idx])})")
                raise ValueError(f"Books to insert contains tuples of wrong length. See printed output above.")

            extras.execute_values(cursor, "INSERT INTO Books (id, ncf_id, goodreads_id, goodreads_url, title, description, publication_year, cover_image_url, average_rating, ratings_count) VALUES %s", books_to_insert, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        except Exception as e:
            # --- Add Enhanced Error Logging ---
            logging.error(f"Error inserting books: {e}")
            # Optionally, try to find the specific row causing the issue (can be slow for large batches)
            # for i, book_tuple in enumerate(books_to_insert):
            #     try:
            #         # Attempt to insert one by one (very inefficient, for debugging only)
            #         # cursor.execute("INSERT INTO Books (...) VALUES (%s, %s, ...)", book_tuple)
            #         # conn.commit() # Or handle transaction differently
            #         pass # Replace with single insert logic if needed
            #     except Exception as row_e:
            #         if "out of range" in str(row_e):
            #              logging.error(f"Row {i} likely caused 'out of range' error: {book_tuple}")
            #              # Log specific problematic values
            #              logging.error(f"  goodreads_id: {book_tuple[2]}, ratings_count: {book_tuple[10]}, pub_year: {book_tuple[6]}")
            #              break # Stop after finding the first error
            # --- End Enhanced Error Logging ---
            conn.rollback(); return
    else:
        logging.warning("No books found to insert.")

    # --- 6. Insert Book-Author Relations ---
    # (Implementation unchanged)
    logging.info(f"Inserting {len(book_author_relations)} book-author relations...")
    if book_author_relations:
      try: extras.execute_values(cursor, "INSERT INTO BookAuthors (book_id, author_id) VALUES %s ON CONFLICT DO NOTHING", book_author_relations, template="(%s, %s)")
      except Exception as e: logging.error(f"Error inserting book-author relations: {e}"); conn.rollback(); return

    # --- 7. Insert Book-Genre Relations ---
    # (Implementation unchanged)
    logging.info("Mapping and Inserting Book-Genre relations...")
    book_genre_relations_to_insert = []
    for _, row in books_df.iterrows():
        goodreads_book_id = row['goodreads_id']
        if goodreads_book_id in book_mapping:
            book_uuid = book_mapping[goodreads_book_id]; genres = extract_genres(row['popular_shelves_list'])
            for genre_name in genres:
                if genre_name in genre_mapping: book_genre_relations_to_insert.append((book_uuid, genre_mapping[genre_name]))
    if book_genre_relations_to_insert:
      logging.info(f"Inserting {len(book_genre_relations_to_insert)} book-genre relations...")
      try: extras.execute_values(cursor, "INSERT INTO BookGenres (book_id, genre_id) VALUES %s ON CONFLICT DO NOTHING", book_genre_relations_to_insert, template="(%s, %s)")
      except Exception as e: logging.error(f"Error inserting book-genre relations: {e}"); conn.rollback(); return

    # --- 8. Insert Similar Book Relations ---
    # (Implementation unchanged)
    logging.info("Mapping and Inserting Similar Book relations...")
    similar_books_to_insert = []
    for gr_id1, gr_id2 in similar_book_relations:
        if gr_id1 in book_mapping and gr_id2 in book_mapping:
            uuid1 = book_mapping[gr_id1]; uuid2 = book_mapping[gr_id2]
            if uuid1 < uuid2: similar_books_to_insert.append((uuid1, uuid2))
            else: similar_books_to_insert.append((uuid2, uuid1))
    if similar_books_to_insert:
        logging.info(f"Inserting {len(similar_books_to_insert)} similar book relations...")
        try: extras.execute_values(cursor, "INSERT INTO SimilarBooks (book_id_1, book_id_2) VALUES %s ON CONFLICT DO NOTHING", similar_books_to_insert, template="(%s, %s)")
        except Exception as e: logging.error(f"Error inserting similar book relations: {e}"); conn.rollback(); return

    # --- 9. Process and Insert Reviews ---
    logging.info("Processing and Inserting Reviews...")
    reviews_to_insert = []; processed_review_ids = set()
    # Remove the .apply() pre-processing for rating here

    reviews_df['review_text'] = reviews_df['review_text'].astype(str).fillna('') # Keep text processing

    for _, row in reviews_df.iterrows():
        goodreads_review_id = row['review_id']
        if pd.isna(goodreads_review_id) or goodreads_review_id in processed_review_ids: continue

        user_id_from_review = row['user_id'] # This is the original_userId UUID string
        book_id_from_review = row['book_id'] # This is the goodreads book ID

        if not is_valid_uuid(user_id_from_review):
            logging.warning(f"Skipping review {goodreads_review_id}: Invalid user UUID format '{user_id_from_review}'.")
            continue

        # Check if user and book exist in our mappings/sets
        if user_id_from_review in all_input_user_ids and book_id_from_review in book_mapping:
            db_book_pk_uuid = book_mapping[book_id_from_review]
            new_review_uuid = uuid.uuid4()

            # --- Process rating directly here ---
            raw_rating = row['rating']
            rating_value = safe_to_int(raw_rating) # Convert to int, None if invalid
            if rating_value is not None and rating_value not in range(1, 6):
                rating_value = None # Set to None if valid int but outside 1-5 range
            # --- End rating processing ---

            # Prepare tuple for insertion
            review_tuple = (
                new_review_uuid,
                row['review_text'],
                rating_value, # Use the processed rating_value
                db_book_pk_uuid,
                user_id_from_review
            )
            reviews_to_insert.append(review_tuple)
            processed_review_ids.add(goodreads_review_id)

    # --- Insertion Logic (remains the same) ---
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
          # --- Add Debugging for the specific error ---
          if "out of range" in str(e):
              # Try to find a problematic tuple (might be slow)
              for i, r_tuple in enumerate(reviews_to_insert):
                  try:
                      # Test individual insert (comment out execute_values above if using this)
                      # cursor.execute("INSERT INTO Reviews (id, text, rating, book_id, user_id) VALUES (%s, %s, %s, %s, %s)", r_tuple)
                      pass # Placeholder if not doing individual inserts
                  except Exception as row_e:
                      if "out of range" in str(row_e):
                          logging.error(f"Row {i} likely caused 'out of range' error for Reviews: {r_tuple}")
                          logging.error(f"  Problematic rating value: {r_tuple[2]}, type: {type(r_tuple[2])}") # Index 2 is rating
                          break # Stop after first error found
          # --- End Debugging ---
          conn.rollback()
          return
    else:
        logging.warning("No valid reviews found/mapped to insert.")


    # --- 10. Process and Insert User Library Entries ---
    # (User ID handling simplified - no mapping needed)
    logging.info("Processing and Inserting User Library entries...")
    library_entries_to_insert = []; processed_library_pairs = set()
    for _, row in interactions_df.iterrows():
        user_id_from_interaction = row['user_id'] # This is the original_userId UUID string
        book_id_from_interaction = row['book_id']
        if not is_valid_uuid(user_id_from_interaction): logging.warning(f"Skipping library entry: Invalid user UUID format '{user_id_from_interaction}' for book {book_id_from_interaction}."); continue
        user_book_pair = (user_id_from_interaction, book_id_from_interaction)
        if user_book_pair in processed_library_pairs: continue
        if user_id_from_interaction in all_input_user_ids and book_id_from_interaction in book_mapping:
            db_book_pk_uuid = book_mapping[book_id_from_interaction]
            added_at_dt = parse_goodreads_date(row['date_added'])
            library_entries_to_insert.append((user_id_from_interaction, db_book_pk_uuid, added_at_dt))
            processed_library_pairs.add(user_book_pair)
    if library_entries_to_insert:
      logging.info(f"Inserting {len(library_entries_to_insert)} user library entries...")
      try: extras.execute_values(cursor, "INSERT INTO UserLibrary (user_id, book_id, added_at) VALUES %s ON CONFLICT (user_id, book_id) DO NOTHING", library_entries_to_insert, template="(%s, %s, %s)")
      except Exception as e: logging.error(f"Error inserting user library entries: {e}"); conn.rollback(); return
    else: logging.warning("No valid user library entries found/mapped to insert.")

    # --- Commit Transaction ---
    conn.commit()
    logging.info("Successfully inserted all data and committed transaction.")

# --- Main Execution Block ---
if __name__ == "__main__":
    conn = None
    try:
        logging.info("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        psycopg2.extras.register_uuid()

        logging.info("Starting data processing...")
        process_and_insert_data(conn, users_df, authors_df, books_df, reviews_df, interactions_df)

    except psycopg2.OperationalError as db_err:
        logging.error(f"Database connection failed: {db_err}")
    except psycopg2.DatabaseError as db_err:
        logging.error(f"Database error during processing: {db_err}")
        if conn: conn.rollback()
    except ValueError as val_err: # Catch config errors and the ambiguity error if it bubbles up
         logging.error(f"Configuration or Value error: {val_err}", exc_info=True) # Log traceback for ValueErrors
         if conn: conn.rollback()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        if conn: conn.rollback()
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")