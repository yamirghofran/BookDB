import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import pandas as pd
import numpy as np
from gmf import GMF
from utils import resume_checkpoint

import os
import glob
import re

# Create output directory if it doesn't exist
output_dir = 'embeddings'
os.makedirs(output_dir, exist_ok=True)

# Copy your GMF config here
gmf_config = {
    'alias': 'gmf_factor32neg4-implict',
    'num_epoch': 20,
    'batch_size': 4096,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'num_users': 205242,
    'num_items': 17663,
    'latent_dim': 32,
    'num_negative': 4,
    'l2_regularization': 0,
    'weight_init_gaussian': True,
    'use_cuda': True,
    'use_bachify_eval': True,
    'device_id': 0,
    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}
# Set device to MPS if available
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using MPS device')
else:
    device = torch.device('cpu')
    print('Using CPU device')

def get_best_checkpoint(model_prefix):
    """Return the checkpoint file path with the best HR (and NDCG as tiebreaker) for a given model prefix."""
    checkpoint_dir = 'checkpoints'
    pattern = os.path.join(checkpoint_dir, f"{model_prefix}*_Epoch*_HR*_NDCG*.model")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found for pattern: {pattern}")

    best_file = None
    best_hr = -1
    best_ndcg = -1

    # Regex to extract HR and NDCG from filename
    regex = re.compile(r'_HR([0-9.]+)_NDCG([0-9.]+)\.model')

    for file in checkpoint_files:
        match = regex.search(file)
        if match:
            hr = float(match.group(1))
            ndcg = float(match.group(2))
            if (hr > best_hr) or (hr == best_hr and ndcg > best_ndcg):
                best_hr = hr
                best_ndcg = ndcg
                best_file = file

    if best_file is None:
        raise ValueError("No valid checkpoint files found with HR and NDCG in the filename.")
    return best_file


# 1. Load best GMF checkpoint
best_checkpoint_path = get_best_checkpoint('gmf')

# 2. Load config and model
config = gmf_config
config['use_cuda'] = False  # Not used here, using device directly
model = GMF(config)
resume_checkpoint(model, best_checkpoint_path, device=device)
model.to(device)

# 3. Extract embeddings
user_embeddings = model.embedding_user.weight.data.to('cpu').numpy()  # (num_users, latent_dim)
item_embeddings = model.embedding_item.weight.data.to('cpu').numpy()  # (num_items, latent_dim)

# 4. Save to parquet
user_ids = np.arange(user_embeddings.shape[0])
item_ids = np.arange(item_embeddings.shape[0])

user_df = pd.DataFrame(user_embeddings)
user_df.insert(0, 'user_id', user_ids)
user_df.to_parquet(os.path.join(output_dir, 'gmf_user_embeddings.parquet'), index=False)

item_df = pd.DataFrame(item_embeddings)
item_df.insert(0, 'item_id', item_ids)
item_df.to_parquet(os.path.join(output_dir, 'gmf_book_embeddings.parquet'), index=False)

print(f'User and item embeddings exported to {output_dir}/gmf_user_embeddings.parquet and {output_dir}/gmf_book_embeddings.parquet')
