import dask.dataframe as dd
import logging
import os
from dask.diagnostics import ProgressBar
from cross_encoder_data_prep import generate_training_data_dask

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Calculate optimal number of partitions based on available CPU cores
    num_cores = os.cpu_count()
    optimal_partitions = max(1, num_cores * 2)  # 2 partitions per core
    logger.info(f"Using {optimal_partitions} partitions based on {num_cores} CPU cores")

    # Load the data
    logger.info("Loading data...")
    user_book_lists = dd.read_parquet("data/sampled_users_book.parquet", engine='pyarrow')
    reduced_books_df = dd.read_parquet("data/reduce_books_df.parquet", engine='pyarrow')
    book_texts_reduced = dd.read_parquet("data/book_texts_reduced.parquet", engine='pyarrow')

    # Generate training data with progress tracking
    logger.info("Starting training data generation...")

    with ProgressBar():
        training_pairs_ddf = generate_training_data_dask(
            user_top_books_ddf=user_book_lists,
            reduced_books_ddf=reduced_books_df,
            reduced_book_texts_ddf=book_texts_reduced,
            output_pairs_path='data/training_pairs.parquet',
            neg_ratio=3,
            topk_books_ctx=3,
            topk_genres_ctx=2,
            dask_partitions=optimal_partitions
        )
        
        # Compute and save results
        logger.info("Computing and saving results...")
        training_pairs_pd = training_pairs_ddf.compute()

    # Display sample of results
    logger.info("Training data generation completed!")
    print("\nSample of generated training pairs:")
    print(training_pairs_pd.head())

if __name__ == "__main__":
    main()