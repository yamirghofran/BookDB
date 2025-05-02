import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import os
import glob

import re

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
              #'num_users': 6040,
              #'num_items': 3706,
              'latent_dim': 32,
              'num_negative': 4,
              'l2_regularization': 0,  # 0.01
              'weight_init_gaussian': True,
              'use_cuda': True,
              'use_bachify_eval': True,
              'device_id': 0,
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor32neg3_bz4096_166432168_pretrain_reg_0.0000001',
              'num_epoch': 20,
              'batch_size': 4096,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              #'num_users': 6040,
              #'num_items': 3706,
              'latent_dim': 32,
              'num_negative': 4,
              'layers': [64, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'weight_init_gaussian': True,
              'use_cuda': True,
              'use_bachify_eval': True,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg3_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'neumf_factor32neg4',
                'num_epoch': 20,
                'batch_size': 4096,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                #'num_users': 6040,
                #'num_items': 3706,
                'latent_dim_mf': 32,
                'latent_dim_mlp': 32,
                'num_negative': 4,
                'layers': [64, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': True,
                'use_bachify_eval': True,
                'device_id': 0,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg3_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg3_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

# Load Data
# ml1m_dir = 'data/ml-1m/ratings.dat'
# ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
# # Reindex
# user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
# user_id['userId'] = np.arange(len(user_id))
# ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
# item_id = ml1m_rating[['mid']].drop_duplicates()
# item_id['itemId'] = np.arange(len(item_id))
# ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
# ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
# print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
# print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# # DataLoader for training
# sample_generator = SampleGenerator(ratings=ml1m_rating)

book_interactions_dir = 'data/books/interactions.parquet'
book_interactions = pd.read_parquet(book_interactions_dir)
print('Range of userId is [{}, {}]'.format(book_interactions.userId.min(), book_interactions.userId.max()))
print('Range of itemId is [{}, {}]'.format(book_interactions.itemId.min(), book_interactions.itemId.max()))
num_users = book_interactions['userId'].max() + 1
num_items = book_interactions['itemId'].max() + 1

gmf_config['num_users'] = num_users
gmf_config['num_items'] = num_items
mlp_config['num_users'] = num_users
mlp_config['num_items'] = num_items
neumf_config['num_users'] = num_users
neumf_config['num_items'] = num_items

if __name__ == '__main__':
    # DataLoader for training
    sample_generator = SampleGenerator(ratings=book_interactions)
    evaluate_data = sample_generator.evaluate_data
    # Specify the exact model
    # config = gmf_config
    # engine = GMFEngine(config)

    # gmf_metrics_history = []

    # for epoch in range(config['num_epoch']):
    #     print('Epoch {} starts !'.format(epoch))
    #     print('-' * 40)
    #     train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    #     engine.train_an_epoch(train_loader, epoch_id=epoch)
    #     hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    #     engine.save(config['alias'], epoch, hit_ratio, ndcg)
    #     gmf_metrics_history.append({
    #         'epoch': epoch,
    #         'hit_ratio': hit_ratio,
    #         'ndcg': ndcg
    #     })

    # # Save metrics history
    # pd.DataFrame(gmf_metrics_history).to_csv('../res/gmf_metrics_history.csv', index=False)


    # config = mlp_config
    # engine = MLPEngine(config)

    # mlp_metrics_history = []

    # for epoch in range(config['num_epoch']):
    #     print('Epoch {} starts !'.format(epoch))
    #     print('-' * 40)
    #     train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    #     engine.train_an_epoch(train_loader, epoch_id=epoch)
    #     hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    #     engine.save(config['alias'], epoch, hit_ratio, ndcg)
    #     mlp_metrics_history.append({
    #         'epoch': epoch,
    #         'hit_ratio': hit_ratio,
    #         'ndcg': ndcg
    #     })

    # # Save metrics history
    # pd.DataFrame(mlp_metrics_history).to_csv('../res/mlp_metrics_history.csv', index=False)

    neumf_metrics_history = []

    # Find best GMF and MLP checkpoints based on HR and NDCG
    best_gmf = get_best_checkpoint('gmf')
    best_mlp = get_best_checkpoint('mlp')

    # Update NeuMF config with best checkpoints
    neumf_config['pretrain_mf'] = best_gmf
    neumf_config['pretrain_mlp'] = best_mlp

    print("Using GMF checkpoint:", best_gmf)
    print("Using MLP checkpoint:", best_mlp)

    config = neumf_config
    engine = NeuMFEngine(config)
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
        neumf_metrics_history.append({
            'epoch': epoch,
            'hit_ratio': hit_ratio,
            'ndcg': ndcg
        })
    # Save metrics history
    pd.DataFrame(neumf_metrics_history).to_csv('../res/neumf_metrics_history.csv', index=False)

