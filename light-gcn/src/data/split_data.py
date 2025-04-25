import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import sys # For printing errors

# Helper function
def _create_edge_index_from_df(df):
    """Converts user_idx, item_idx columns of a DataFrame to a PyTorch edge_index tensor."""
    if df.empty:
        return torch.empty((2, 0), dtype=torch.long)
    if not all(col in df.columns for col in ['user_idx', 'item_idx']):
         raise ValueError("DataFrame must contain 'user_idx' and 'item_idx' columns.")
         
    user_indices = df['user_idx'].to_numpy()
    item_indices = df['item_idx'].to_numpy()
    user_tensor = torch.tensor(user_indices, dtype=torch.long)
    item_tensor = torch.tensor(item_indices, dtype=torch.long)
    return torch.stack([user_tensor, item_tensor], dim=0)

# Main splitting function
def split_data_per_user(positive_df, test_ratio=0.15, val_ratio=0.15, random_state=42):
    """
    Splits positive interaction data per user into train, validation, and test sets.

    Ensures that users with few interactions are handled gracefully.

    Args:
        positive_df (pd.DataFrame): DataFrame containing positive interactions. 
                                     Must include 'user_idx' and 'item_idx' columns.
        test_ratio (float): Desired proportion of original data for the test set (e.g., 0.15 for 15%).
        val_ratio (float): Desired proportion of original data for the validation set (e.g., 0.15 for 15%).
        random_state (int): Random state for reproducible splits.

    Returns:
        tuple: A tuple containing three torch.Tensor objects:
               (train_edge_index, val_edge_index, test_edge_index)
               Returns empty tensors if the input DataFrame is empty.
    """
    print(f"--- Starting Data Split (Target: Train={1-(test_ratio+val_ratio):.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}) ---")

    # --- Input Validation ---
    if not isinstance(positive_df, pd.DataFrame):
        raise TypeError("Input 'positive_df' must be a pandas DataFrame.")
    if not all(col in positive_df.columns for col in ['user_idx', 'item_idx']):
        raise ValueError("Input DataFrame must contain 'user_idx' and 'item_idx' columns.")
    if not (0 < test_ratio < 1 and 0 < val_ratio < 1 and (test_ratio + val_ratio) < 1):
         raise ValueError("test_ratio and val_ratio must be between 0 and 1, and their sum must be less than 1.")
         
    if positive_df.empty:
        print("Warning: Input DataFrame is empty. Returning empty edge indices.")
        empty_idx = torch.empty((2, 0), dtype=torch.long)
        return empty_idx, empty_idx, empty_idx
        
    original_total = len(positive_df)
    print(f"Input DataFrame contains {original_total} positive interactions.")

    # --- Step 1: Split into Train+Validation vs. Test (per user) ---
    test_interactions_list = []
    train_val_interactions_list = []
    
    grouped_interactions = positive_df.groupby('user_idx')
    print(f"Processing {grouped_interactions.ngroups} users for initial Train+Val/Test split...")

    for user_idx, group in grouped_interactions:
        n_interactions = len(group)
        
        if n_interactions < 3: 
            # Cannot split into 3 sets, put all in train_val for now
            train_val_interactions_list.append(group)
            continue
            
        # Calculate number of test samples
        n_test_samples = max(1, int(np.round(n_interactions * test_ratio))) # Ensure at least 1
        
        # Ensure at least 2 interactions remain for train/val split
        if n_interactions - n_test_samples < 2:
            if n_interactions == 3: n_test_samples = 1 # Leave 2 for train/val
            elif n_interactions == 2: n_test_samples = 0 # Cannot take test if only 2 exist initially
            # (len < 2 already handled by initial continue)

        if n_test_samples > 0:
            train_val_group, test_group = train_test_split(group, test_size=n_test_samples, random_state=random_state)
            test_interactions_list.append(test_group)
            train_val_interactions_list.append(train_val_group)
        else:
             # Keep all in train_val if no test samples could be taken
             train_val_interactions_list.append(group)


    # Concatenate the lists back into DataFrames
    # Handle cases where one list might be empty
    train_val_df = pd.concat(train_val_interactions_list).reset_index(drop=True) if train_val_interactions_list else pd.DataFrame(columns=positive_df.columns)
    test_df = pd.concat(test_interactions_list).reset_index(drop=True) if test_interactions_list else pd.DataFrame(columns=positive_df.columns)
    
    print(f"Initial split results: {len(train_val_df)} train+val interactions, {len(test_df)} test interactions.")
    if len(train_val_df) + len(test_df) != original_total:
         print(f"Warning: Interaction count mismatch after test split ({len(train_val_df) + len(test_df)} vs {original_total}). Check logic.", file=sys.stderr)

    # --- Step 2: Split Train+Validation into Train vs. Validation (per user) ---
    val_interactions_list = []
    train_interactions_list = []

    # Calculate relative validation proportion needed from the train_val set
    train_val_total = len(train_val_df)
    if train_val_total == 0: # Handle empty train_val_df
        print("Warning: train_val_df is empty after test split. No validation set possible.")
        train_df = train_val_df # Empty df
        val_df = train_val_df # Empty df
    else:
        # Calculate target validation size and relative proportion
        target_val_size = int(np.round(original_total * val_ratio))
        # Calculate proportion relative to current train_val set size
        if (1.0 - test_ratio) <= 1e-6: # Avoid division by zero or near-zero
             val_size_proportion_relative = 0 
        else:
             val_size_proportion_relative = val_ratio / (1.0 - test_ratio)
        
        grouped_train_val = train_val_df.groupby('user_idx')
        print(f"Processing {grouped_train_val.ngroups} users for Train/Validation split (target val size: {target_val_size})...")

        for user_idx, group in grouped_train_val:
            n_interactions = len(group)
            
            if n_interactions < 2:
                # If only 1 interaction left, must go to train
                train_interactions_list.append(group)
                continue
                
            # Calculate number of validation samples
            n_val_samples = max(1, int(np.round(n_interactions * val_size_proportion_relative))) 

            # Ensure at least 1 interaction remains for training
            if n_interactions - n_val_samples < 1:
                 n_val_samples = n_interactions - 1 # Leave exactly one for training
            
            if n_val_samples > 0 :
                train_group, val_group = train_test_split(group, test_size=n_val_samples, random_state=random_state)
                val_interactions_list.append(val_group)
                train_interactions_list.append(train_group)
            else:
                # Keep all in train if no validation samples could be taken
                train_interactions_list.append(group)


        # Concatenate the lists back into DataFrames
        train_df = pd.concat(train_interactions_list).reset_index(drop=True) if train_interactions_list else pd.DataFrame(columns=positive_df.columns)
        val_df = pd.concat(val_interactions_list).reset_index(drop=True) if val_interactions_list else pd.DataFrame(columns=positive_df.columns)

    # --- Report Final Counts ---
    print(f"\nFinal split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test interactions.")
    total_split = len(train_df) + len(val_df) + len(test_df)
    print(f"Total interactions in splits: {total_split} (Original positive: {original_total})")
    if total_split != original_total:
        print("Warning: Mismatch in total interactions after split - check logic for small groups.", file=sys.stderr)

    # Calculate percentages
    train_perc = len(train_df) / original_total * 100 if original_total > 0 else 0
    val_perc = len(val_df) / original_total * 100 if original_total > 0 else 0
    test_perc = len(test_df) / original_total * 100 if original_total > 0 else 0
    print(f"Split percentages: Train={train_perc:.2f}%, Validation={val_perc:.2f}%, Test={test_perc:.2f}%")

    # --- Step 3: Create edge_index tensors for each set ---
    print("\nConverting splits to edge_index tensors...")
    train_edge_index = _create_edge_index_from_df(train_df)
    val_edge_index = _create_edge_index_from_df(val_df)
    test_edge_index = _create_edge_index_from_df(test_df)

    print(f"Train edge_index shape: {train_edge_index.shape}")
    print(f"Validation edge_index shape: {val_edge_index.shape}")
    print(f"Test edge_index shape: {test_edge_index.shape}")
    print("--- Data Split Complete ---")

    return train_edge_index, val_edge_index, test_edge_index
