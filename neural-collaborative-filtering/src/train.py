import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
import os
from utils import get_best_checkpoint


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

def run_ncf_pipeline(book_interactions, output_dir='../res'):
    """Run the full NCF pipeline (GMF, MLP, NeuMF) and save metrics to output_dir."""
    from data import SampleGenerator
    import pandas as pd

    sample_generator = SampleGenerator(ratings=book_interactions)
    evaluate_data = sample_generator.evaluate_data

    # GMF
    gmf_metrics_history = []
    config = gmf_config
    engine = GMFEngine(config)
    for epoch in range(config['num_epoch']):
        print(f'GMF Epoch {epoch} starts !')
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
        gmf_metrics_history.append({'epoch': epoch, 'hit_ratio': hit_ratio, 'ndcg': ndcg})
    pd.DataFrame(gmf_metrics_history).to_csv(f'{output_dir}/gmf_metrics_history.csv', index=False)

    # MLP
    mlp_metrics_history = []
    config = mlp_config
    engine = MLPEngine(config)
    for epoch in range(config['num_epoch']):
        print(f'MLP Epoch {epoch} starts !')
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
        mlp_metrics_history.append({'epoch': epoch, 'hit_ratio': hit_ratio, 'ndcg': ndcg})
    pd.DataFrame(mlp_metrics_history).to_csv(f'{output_dir}/mlp_metrics_history.csv', index=False)

    # NeuMF
    neumf_metrics_history = []
    best_gmf = get_best_checkpoint('gmf')
    best_mlp = get_best_checkpoint('mlp')
    neumf_config['pretrain_mf'] = best_gmf
    neumf_config['pretrain_mlp'] = best_mlp
    print('Using GMF checkpoint:', best_gmf)
    print('Using MLP checkpoint:', best_mlp)
    config = neumf_config
    engine = NeuMFEngine(config)
    for epoch in range(config['num_epoch']):
        print(f'NeuMF Epoch {epoch} starts !')
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
        neumf_metrics_history.append({'epoch': epoch, 'hit_ratio': hit_ratio, 'ndcg': ndcg})
    pd.DataFrame(neumf_metrics_history).to_csv(f'{output_dir}/neumf_metrics_history.csv', index=False)

book_interactions_dir = 'data/interactions_prepared_ncf_reduced.parquet'
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

# if __name__ == '__main__':
#     run_ncf_pipeline(book_interactions)
