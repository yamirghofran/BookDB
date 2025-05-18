import pandas as pd
import dask.dataframe as dd
import random
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_book_info(book_id: str, reduced_books_df: pd.DataFrame) -> dict:
    """
    Retrieve book metadata for a given book_id from a pandas DataFrame.
    Assumes reduced_books_df is a pandas DataFrame.
    Args:
        book_id: The ID of the book to look up
        reduced_books_df: pandas DataFrame containing book metadata
        
    Returns:
        dict: Book metadata including title, authors, and genres
        None: If book is not found
    """
    try:
        book_row = reduced_books_df[reduced_books_df['book_id'] == book_id]
        if book_row.empty:
            return None
        book_row = book_row.iloc[0]
        return {
            'title': book_row['title'],
            'authors': book_row['authors'].split(',') if pd.notna(book_row['authors']) else [],
            'genres': book_row['genre'].split(',') if pd.notna(book_row['genre']) else []
        }
    except IndexError:
        return None
    except Exception as e:
        logger.error(f"Error retrieving book info for {book_id} (pandas df): {str(e)}")
        return None

def make_user_context(user_id: str,
                      target_book: str,
                      user_top_books_list: list, 
                      book_meta_pd: pd.DataFrame, 
                      topk_books: int = 3,
                      topk_genres: int = 2) -> str:
    """
    Build a leave-one-out user context string excluding the target book.
    Keeps looking through user's book list until it finds valid books for context.
    """
    try:
        # 1. Get all books excluding target book
        favorites = [b for b in user_top_books_list if b != target_book]
        if not favorites:
            # If all books are target book, use target book info
            favorites = [target_book]
        
        # 2. Process books until we find enough valid ones
        book_strings = []
        all_genres = []
        books_processed = 0
        
        # Keep going through the list until we have enough books or run out
        while len(book_strings) < topk_books and books_processed < len(favorites):
            current_book = favorites[books_processed]
            info = get_book_info(current_book, book_meta_pd)
            
            if info:  # Only add if we found valid book info
                authors = " and ".join(info['authors']) if info['authors'] else "Unknown Author"
                title = info['title'] if info['title'] else f"Book {current_book}"
                book_strings.append(f"{title} by {authors}")
                
                if info['genres']:
                    all_genres.extend(info['genres'])
            
            books_processed += 1

        # 3. Create favorite books string
        fav_str = ""
        if book_strings:
            fav_str = book_strings[0] if len(book_strings) == 1 else ", ".join(book_strings[:-1]) + " and " + book_strings[-1]
        else:
            # Fallback if no book strings could be created
            fav_str = f"User has read {len(user_top_books_list)} books"
        
        # 4. Create genres string
        genre_counts = Counter(all_genres)
        top_genres_list = [g for g, _ in genre_counts.most_common(topk_genres)]
        genre_str = " and ".join(top_genres_list) if top_genres_list else "various genres"

        # 5. Combine parts
        parts = []
        if fav_str: parts.append(f"Favorite books: {fav_str}.")
        if genre_str: parts.append(f"Favorite genres: {genre_str}.")
        
        # 6. Ensure we always return a non-empty context
        if not parts:
            return f"User has read {len(user_top_books_list)} books."
            
        return " ".join(parts)
    except Exception as e:
        logger.warning(f"Ctx error for user {user_id}, target {target_book}: {str(e)}. Using fallback context.")
        return f"User has read {len(user_top_books_list)} books."

def create_training_pairs(
    user_top_books_df: pd.DataFrame,
    reduced_books_df: pd.DataFrame,
    reduced_book_texts_df: pd.DataFrame,
    neg_ratio: int = 3,
    topk_books_ctx: int = 3,
    topk_genres_ctx: int = 2
) -> pd.DataFrame:
    """Pandas-based training pair generation."""
    try:
        # Validations (assuming they are done correctly elsewhere or simplified for brevity)
        user2top_dict = user_top_books_df.set_index('user_id')['books_read'].to_dict()
        all_book_ids_set = set(reduced_books_df['book_id'].unique())
        book_texts_dict = reduced_book_texts_df.set_index('book_id')['text'].to_dict()
        records = []

        for u, user_books_list in tqdm(user2top_dict.items(), desc="Creating training pairs (Pandas)"):
            if not user_books_list: continue
            for b_pos in user_books_list:
                if b_pos not in all_book_ids_set: continue
                ctx = make_user_context(u, b_pos, user_books_list, reduced_books_df, topk_books_ctx, topk_genres_ctx)
                book_text = book_texts_dict.get(b_pos)
                if book_text is None: continue
                records.append({'user_id': u, 'user_ctx': ctx, 'book_id': b_pos, 'book_text': book_text, 'label': 1})
                
                neg_pool = list(all_book_ids_set - set(user_books_list))
                if not neg_pool: continue
                actual_neg_ratio = min(neg_ratio, len(neg_pool))
                if actual_neg_ratio > 0:
                    negs = random.sample(neg_pool, actual_neg_ratio)
                    for b_neg in negs:
                        ctx_neg = make_user_context(u, b_neg, user_books_list, reduced_books_df, topk_books_ctx, topk_genres_ctx)
                        book_text_neg = book_texts_dict.get(b_neg)
                        if book_text_neg is None: continue
                        records.append({'user_id': u, 'user_ctx': ctx_neg, 'book_id': b_neg, 'book_text': book_text_neg, 'label': 0})
        
        if not records: return pd.DataFrame(columns=['user_id', 'user_ctx', 'book_id', 'book_text', 'label'])
        return pd.DataFrame.from_records(records)
    except Exception as e:
        logger.error(f"Error in create_training_pairs (Pandas): {str(e)}", exc_info=True)
        raise

def _process_dask_partition(
    partition_df: pd.DataFrame,
    book_meta_pd: pd.DataFrame,
    book_texts_dict: dict,
    all_book_ids_set: set,
    neg_ratio: int = 3,
    topk_books_ctx: int = 3,
    topk_genres_ctx: int = 2
) -> pd.DataFrame:
    """
    Process a partition of user data to generate training pairs.
    Optimized for memory usage and includes progress tracking.
    """
    pairs = []
    total_users = len(partition_df)
    
    for idx, row in enumerate(partition_df.itertuples(), 1):
        if idx % 100 == 0:  # Log progress every 100 users
            logger.info(f"Processing user {idx}/{total_users} in partition")
            
        user_id = row.user_id
        user_top_books = row.top_books
        
        # Skip users with no books
        if not user_top_books:
            continue
            
        # Process each book in user's top books
        for target_book in user_top_books:
            # Skip if book not in our dataset
            if target_book not in all_book_ids_set:
                continue
                
            # Get book text
            book_text = book_texts_dict.get(target_book)
            if not book_text:
                continue
                
            # Generate user context
            user_ctx = make_user_context(
                user_id=user_id,
                target_book=target_book,
                user_top_books_list=user_top_books,
                book_meta_pd=book_meta_pd,
                topk_books=topk_books_ctx,
                topk_genres=topk_genres_ctx
            )
            
            # Add positive pair
            pairs.append({
                'user_id': user_id,
                'book_id': target_book,
                'user_ctx': user_ctx,
                'book_text': book_text,
                'label': 1
            })
            
            # Generate negative samples
            neg_books = generate_negative_samples(
                user_top_books=user_top_books,
                all_book_ids_set=all_book_ids_set,
                neg_ratio=neg_ratio
            )
            
            # Add negative pairs
            for neg_book in neg_books:
                neg_text = book_texts_dict.get(neg_book)
                if neg_text:
                    pairs.append({
                        'user_id': user_id,
                        'book_id': neg_book,
                        'user_ctx': user_ctx,
                        'book_text': neg_text,
                        'label': 0
                    })
    
    # Convert to DataFrame
    if pairs:
        return pd.DataFrame(pairs)
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['user_id', 'book_id', 'user_ctx', 'book_text', 'label'])

def create_training_pairs_dask(
    user_top_books_ddf: dd.DataFrame, 
    reduced_books_ddf: dd.DataFrame,
    reduced_book_texts_ddf: dd.DataFrame,
    neg_ratio: int = 3,
    topk_books_ctx: int = 3, 
    topk_genres_ctx: int = 2 
) -> dd.DataFrame:
    """Dask-native training pair generation."""
    logger.info("Starting Dask-native training pair generation...")
    book_meta_pd = reduced_books_ddf.compute()
    book_texts_pd = reduced_book_texts_ddf.compute()
    book_texts_dict = book_texts_pd.set_index('book_id')['text'].to_dict()
    all_book_ids_set = set(book_meta_pd['book_id'].unique())
    logger.info(f"Book metadata (pandas): {book_meta_pd.shape}, Texts dict: {len(book_texts_dict)} entries.")

    user_id_dtype = user_top_books_ddf['user_id'].dtype if 'user_id' in user_top_books_ddf.columns else object
    book_id_dtype = book_meta_pd['book_id'].dtype if 'book_id' in book_meta_pd.columns else object
    meta_df = pd.DataFrame({
        'user_id': pd.Series(dtype=user_id_dtype),
        'user_ctx': pd.Series(dtype=str),
        'book_id': pd.Series(dtype=book_id_dtype),
        'book_text': pd.Series(dtype=str),
        'label': pd.Series(dtype=int)
    })

    training_pairs_ddf = user_top_books_ddf.map_partitions(
        _process_dask_partition,
        book_meta_pd=book_meta_pd,
        book_texts_dict=book_texts_dict,
        all_book_ids_set=all_book_ids_set,
        neg_ratio=neg_ratio,
        topk_books_ctx=topk_books_ctx,
        topk_genres_ctx=topk_genres_ctx,
        meta=meta_df
    )
    logger.info("Dask map_partitions task graph created.")
    return training_pairs_ddf

def generate_training_data_from_dataframes(
    user_top_books_df: pd.DataFrame,
    reduced_books_df: pd.DataFrame,
    reduced_book_texts_df: pd.DataFrame,
    output_pairs_path: str,
    neg_ratio: int = 3,
    topk_books_ctx: int = 3,
    topk_genres_ctx: int = 2
) -> pd.DataFrame:
    """Wrapper for pandas DataFrames input and output."""
    logger.info("Starting training pair generation from PANDAS DataFrames...")
    try:
        # Basic DataFrame type validation
        for df, name in [(user_top_books_df, "user_top_books_df"), 
                         (reduced_books_df, "reduced_books_df"), 
                         (reduced_book_texts_df, "reduced_book_texts_df")]:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame, got {type(df)}")

        df_pairs = create_training_pairs(
            user_top_books_df, reduced_books_df, reduced_book_texts_df,
            neg_ratio, topk_books_ctx, topk_genres_ctx
        )
        if df_pairs.empty:
            logger.warning("create_training_pairs (Pandas) returned empty. No file saved.")
            return df_pairs
        Path(output_pairs_path).parent.mkdir(parents=True, exist_ok=True)
        df_pairs.to_parquet(str(output_pairs_path), index=False)
        logger.info(f"Training pairs saved to {output_pairs_path} from PANDAS DataFrames.")
        return df_pairs
    except Exception as e:
        logger.error(f"Error in generate_training_data_from_dataframes: {str(e)}", exc_info=True)
        raise 

def generate_training_data_from_files(
    user_books_path: str, reduced_books_path: str, reduced_texts_path: str,
    output_pairs_path: str, neg_ratio: int = 3,
    topk_books_ctx: int = 3, topk_genres_ctx: int = 2
) -> pd.DataFrame:
    """Wrapper for file inputs, processes via pandas method."""
    logger.info("Generating from files (Dask load -> Pandas compute & process)..." )
    try:
        user_top_books_df = dd.read_parquet(user_books_path).compute()
        reduced_books_df = dd.read_parquet(reduced_books_path).compute()
        reduced_book_texts_df = dd.read_parquet(reduced_texts_path).compute()
        return generate_training_data_from_dataframes(
            user_top_books_df, reduced_books_df, reduced_book_texts_df,
            output_pairs_path, neg_ratio, topk_books_ctx, topk_genres_ctx
        )
    except Exception as e:
        logger.error(f"Error in generate_training_data_from_files: {str(e)}", exc_info=True)
        raise

def generate_training_data_dask(
    user_top_books_ddf: dd.DataFrame,
    reduced_books_ddf: dd.DataFrame,
    reduced_book_texts_ddf: dd.DataFrame,
    output_pairs_path: str,
    neg_ratio: int = 3,
    topk_books_ctx: int = 3,
    topk_genres_ctx: int = 2,
    dask_partitions: int = -1 
) -> dd.DataFrame:
    """
    Generate training pairs using Dask for distributed processing.
    Includes progress tracking and memory optimization.
    """
    logger.info("Starting Dask-native generation and saving...")
    
    # 1. Compute and cache the book metadata and texts
    logger.info("Computing and caching book metadata...")
    book_meta_pd = reduced_books_ddf.compute()
    logger.info("Computing and caching book texts...")
    book_texts_pd = reduced_book_texts_ddf.compute()
    
    # Convert to dictionaries for faster lookups
    book_texts_dict = dict(zip(book_texts_pd['book_id'], book_texts_pd['text']))
    all_book_ids_set = set(book_meta_pd['book_id'].unique())
    
    # 2. Repartition if needed
    if dask_partitions > 0:
        user_top_books_ddf = user_top_books_ddf.repartition(npartitions=dask_partitions)
    
    # 3. Create the processing function with progress tracking
    def process_with_progress(partition_df):
        partition_size = len(partition_df)
        logger.info(f"Processing partition with {partition_size} users...")
        
        result = _process_dask_partition(
            partition_df=partition_df,
            book_meta_pd=book_meta_pd,
            book_texts_dict=book_texts_dict,
            all_book_ids_set=all_book_ids_set,
            neg_ratio=neg_ratio,
            topk_books_ctx=topk_books_ctx,
            topk_genres_ctx=topk_genres_ctx
        )
        
        logger.info(f"Completed partition with {len(result)} pairs generated")
        return result
    
    # 4. Apply the processing with progress tracking
    logger.info("Starting Dask-native training pair generation...")
    training_pairs_ddf = user_top_books_ddf.map_partitions(
        process_with_progress,
        meta={
            'user_id': 'object',
            'book_id': 'object',
            'user_ctx': 'object',
            'book_text': 'object',
            'label': 'int64'
        }
    )
    
    # 5. Save the results with progress tracking
    logger.info("Saving results to parquet file...")
    training_pairs_ddf.to_parquet(
        output_pairs_path,
        engine='pyarrow',
        compression='snappy',
        write_index=False
    )
    
    logger.info("Training data generation completed!")
    return training_pairs_ddf

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build cross-encoder training pairs')
    parser.add_argument('--user_books', required=True)
    parser.add_argument('--reduced_books', required=True)
    parser.add_argument('--reduced_texts', required=True)
    parser.add_argument('--output_pairs', required=True)
    parser.add_argument('--neg_ratio', type=int, default=3)
    parser.add_argument('--topk_books_ctx', type=int, default=3)
    parser.add_argument('--topk_genres_ctx', type=int, default=2)
    parser.add_argument('--processing_mode', default='pandas', choices=['pandas', 'dask'])
    parser.add_argument('--dask_output_partitions', type=int, default=-1)
    args = parser.parse_args()

    try:
        if args.processing_mode == 'dask':
            logger.info("CLI: Dask mode selected.")
            generate_training_data_dask(
                dd.read_parquet(args.user_books),
                dd.read_parquet(args.reduced_books),
                dd.read_parquet(args.reduced_texts),
                args.output_pairs,
                args.neg_ratio, args.topk_books_ctx, args.topk_genres_ctx,
                args.dask_output_partitions
            )
        else:
            logger.info("CLI: Pandas mode selected.")
            generate_training_data_from_files(
                args.user_books, args.reduced_books, args.reduced_texts,
                args.output_pairs, args.neg_ratio, 
                args.topk_books_ctx, args.topk_genres_ctx
            )
        logger.info("CLI: Process completed.")
    except Exception as e:
        logger.error(f"CLI Error: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()
