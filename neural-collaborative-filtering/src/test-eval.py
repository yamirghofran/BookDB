from data import SampleGenerator
from gmf import GMFEngine  # or MLPEngine, NeuMFEngine
import pandas as pd
import torch

gmf_config = {'alias': 'gmf_factor32neg4-implict',
              'num_epoch': 20,
              'batch_size': 4096,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 32,
              'num_negative': 4,
              'l2_regularization': 0,  # 0.01
              'weight_init_gaussian': True,
              'use_cuda': True,
              'use_bachify_eval': False,
              'device_id': 0,
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}


book_interactions_dir = 'data/books/interactions.parquet'
book_interactions = pd.read_parquet(book_interactions_dir)
print('Range of userId is [{}, {}]'.format(book_interactions.userId.min(), book_interactions.userId.max()))
print('Range of itemId is [{}, {}]'.format(book_interactions.itemId.min(), book_interactions.itemId.max()))
num_users = book_interactions['userId'].max() + 1
num_items = book_interactions['itemId'].max() + 1

gmf_config['num_users'] = num_users
gmf_config['num_items'] = num_items

# Assume you have ratings data loaded as a DataFrame called book_interactions
sample_generator = SampleGenerator(ratings=book_interactions)

# Use your config (e.g., gmf_config)
engine = GMFEngine(gmf_config)

# Get evaluation data (you can use fewer negatives for a quick test)
evaluate_data = sample_generator.get_evaluation_with_negatives(num_negatives=3)

# Call evaluate (epoch_id can be 0 or any integer)
hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=0)

print("Hit Ratio:", hit_ratio)
print("NDCG:", ndcg)

def get_ranked_recommendations(engine, user_id, num_recommendations=10, exclude_interacted=True):
    """
    Generate ranked recommendations for a given user.
    
    Args:
        engine: The trained GMF engine
        user_id: The ID of the user to get recommendations for
        num_recommendations: Number of recommendations to return
        exclude_interacted: Whether to exclude items the user has already interacted with
        
    Returns:
        List of (item_id, score) tuples sorted by score in descending order
    """
    # Get user embedding
    user_embedding = engine.model.user_embedding(torch.LongTensor([user_id]).to(engine.config['device_id']))
    
    # Get all item embeddings
    all_items = torch.arange(engine.config['num_items']).to(engine.config['device_id'])
    item_embeddings = engine.model.item_embedding(all_items)
    
    # Compute scores for all items
    scores = torch.mul(user_embedding, item_embeddings).sum(dim=1)
    
    # If excluding interacted items, get user's interaction history
    if exclude_interacted:
        # Load interactions data
        interactions = pd.read_parquet('data/interactions_prepared_ncf_reduced.parquet')
        user_interactions = interactions[interactions['userId'] == user_id]['itemId'].values
        
        # Set scores for interacted items to negative infinity
        scores[user_interactions] = float('-inf')
    
    # Get top N items
    top_scores, top_indices = torch.topk(scores, num_recommendations)
    
    # Convert to list of (item_id, score) tuples
    recommendations = list(zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()))
    
    return recommendations

# Example usage:
# recommendations = get_ranked_recommendations(engine, user_id=1)
# for item_id, score in recommendations:
#     print(f"Item {item_id}: Score {score:.4f}")