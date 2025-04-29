from data import SampleGenerator
from gmf import GMFEngine  # or MLPEngine, NeuMFEngine
import pandas as pd

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