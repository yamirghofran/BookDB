import pandas as pd
import random
import glob
import os

def process_individual_chunk(df_chunk, min_pos_to_keep, max_pos_to_sample, negatives_per_positive_sample, random_seed_value):
    """
    Applies the user-defined sampling logic to a single DataFrame chunk.
    """
    
    # --- Step 1: Identify positive and negative interactions in the chunk ---
    df_positives_chunk = df_chunk[df_chunk['label'] == 1]
    df_negatives_chunk = df_chunk[df_chunk['label'] == 0]

    # --- Step 2: Process each user within the chunk for positive samples ---
    selected_positive_samples_for_chunk = []
    users_in_chunk_positives = df_positives_chunk['user_id'].unique()

    for user_id_val in users_in_chunk_positives:
        user_chunk_positive_df = df_positives_chunk[df_positives_chunk['user_id'] == user_id_val]
        num_user_chunk_positives = len(user_chunk_positive_df)

        if num_user_chunk_positives >= max_pos_to_sample:
            # If >= max_pos_to_sample positives, sample max_pos_to_sample
            selected_positive_samples_for_chunk.append(
                user_chunk_positive_df.sample(n=max_pos_to_sample, random_state=random_seed_value)
            )
        elif num_user_chunk_positives >= min_pos_to_keep:
            # If between min_pos_to_keep and max_pos_to_sample (exclusive for upper), keep all
            selected_positive_samples_for_chunk.append(user_chunk_positive_df)
        # Else (less than min_pos_to_keep positives), drop the user's positives from this chunk

    if not selected_positive_samples_for_chunk:
        # Return an empty DataFrame with the same columns if no positives selected
        return pd.DataFrame(columns=df_chunk.columns)

    final_positives_df_for_chunk = pd.concat(selected_positive_samples_for_chunk).reset_index(drop=True)

    # --- Step 3: Sample negatives for each selected positive in the chunk ---
    final_samples_list_for_chunk = []
    if not final_positives_df_for_chunk.empty:
        for _, positive_row_item in final_positives_df_for_chunk.iterrows():
            user_id_for_neg_sampling = positive_row_item['user_id']
            
            # Add the positive sample
            final_samples_list_for_chunk.append(positive_row_item.to_dict())
            
            # Get all negative samples for this user from the current chunk
            user_chunk_negative_df = df_negatives_chunk[df_negatives_chunk['user_id'] == user_id_for_neg_sampling]
            
            if not user_chunk_negative_df.empty:
                num_negs_to_sample_val = min(negatives_per_positive_sample, len(user_chunk_negative_df))
                if num_negs_to_sample_val > 0:
                    sampled_negatives_for_chunk = user_chunk_negative_df.sample(
                        n=num_negs_to_sample_val, random_state=random_seed_value
                    )
                    for _, neg_row_item in sampled_negatives_for_chunk.iterrows():
                        final_samples_list_for_chunk.append(neg_row_item.to_dict())

    if not final_samples_list_for_chunk:
         return pd.DataFrame(columns=df_chunk.columns) # Return empty if list is empty

    processed_df_output_chunk = pd.DataFrame(final_samples_list_for_chunk)

    if not processed_df_output_chunk.empty:
        # Ensure correct dtypes, especially for label
        processed_df_output_chunk['label'] = processed_df_output_chunk['label'].astype(int)
    
    return processed_df_output_chunk

# --- Main script execution ---
if __name__ == "__main__":
    # Configuration (based on your provided code)
    BASE_DATA_PATH = 'data/training_pairs.parquet/' # Directory containing part.*.parquet files
    
    # --- Where the final dataset is saved ---
    FINAL_OUTPUT_FILE = 'data/processed_training_pairs_parts_0_to_12.parquet' 
    # You can change this path and filename if you wish.
    # For example: 'data/custom_name_for_processed_data.parquet'
    # It will be saved in the 'data' directory relative to where you run the script.
    # If the 'data' directory doesn't exist, the script will try to create it.

    MIN_POSITIVES_TO_KEEP_USER = 3
    MAX_POSITIVES_TO_SAMPLE = 3
    NEGATIVES_PER_POSITIVE = 3
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED) 

    # --- Specify exactly parts 0 to 12 ---
    parts_to_load_indices = range(13) # 0 to 12 inclusive
    parquet_file_paths = [
        os.path.join(BASE_DATA_PATH, f'part.{i}.parquet') for i in parts_to_load_indices
    ]
    
    print(f"Attempting to process {len(parquet_file_paths)} specific Parquet files (parts 0-12) from '{BASE_DATA_PATH}'.")
    print(f"The final processed data will be saved to: {FINAL_OUTPUT_FILE}")


    if not parquet_file_paths: # Should not happen with explicit list unless range is empty
        print(f"No Parquet files specified. Please check the 'parts_to_load_indices'.")
    else:
        print(f"Will attempt to process {len(parquet_file_paths)} specific Parquet files (parts 0-12) from '{BASE_DATA_PATH}'.")
        print(f"The final processed data will be saved to: {FINAL_OUTPUT_FILE}")


    all_processed_dataframes = []

    for i, current_file_path in enumerate(parquet_file_paths):
        print(f"\nProcessing file {i+1}/{len(parquet_file_paths)}: {current_file_path}...")
        try:
            # Check if the specific part file exists before trying to read
            if not os.path.exists(current_file_path):
                print(f"  Warning: File {current_file_path} not found. Skipping.")
                continue

            current_df_chunk = pd.read_parquet(current_file_path)
            print(f"  Loaded chunk with shape: {current_df_chunk.shape}")

            if current_df_chunk.empty:
                print(f"  Chunk from {current_file_path} is empty. Skipping.")
                continue
            
            required_cols = ['user_id', 'label'] 
            if not all(col in current_df_chunk.columns for col in required_cols):
                print(f"  Chunk {current_file_path} is missing one or more required columns: {required_cols}. Skipping.")
                continue

            processed_result_chunk = process_individual_chunk(
                current_df_chunk,
                MIN_POSITIVES_TO_KEEP_USER,
                MAX_POSITIVES_TO_SAMPLE,
                NEGATIVES_PER_POSITIVE,
                RANDOM_SEED
            )
            
            if not processed_result_chunk.empty:
                all_processed_dataframes.append(processed_result_chunk)
                print(f"  Processed chunk shape: {processed_result_chunk.shape}. Result added.")
            else:
                print(f"  Processing of chunk {current_file_path} resulted in an empty DataFrame.")
                
        except FileNotFoundError: # Should be caught by os.path.exists now, but good to keep
            print(f"  Error: File not found {current_file_path}. Skipping.")
        except Exception as e:
            print(f"  Error processing file {current_file_path}: {e}")

    if all_processed_dataframes:
        final_combined_df = pd.concat(all_processed_dataframes, ignore_index=True)
        print(f"\nSuccessfully combined all processed chunks.")
        print(f"Final combined DataFrame shape: {final_combined_df.shape}")
        
        if 'label' in final_combined_df.columns:
             print(f"Label distribution in final DataFrame:\n{final_combined_df['label'].value_counts(normalize=True)}")
        if 'user_id' in final_combined_df.columns:
            print(f"Number of unique users in final DataFrame: {final_combined_df['user_id'].nunique()}")

        output_directory = os.path.dirname(FINAL_OUTPUT_FILE)
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created output directory: {output_directory}")

        final_combined_df.to_parquet(FINAL_OUTPUT_FILE, index=False)
        print(f"Successfully saved combined processed data to {FINAL_OUTPUT_FILE}")
    else:
        print("\nNo data was processed from any of the chunks. Final DataFrame is empty. Nothing saved.")
