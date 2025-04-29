import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


import torch
import random
import pandas as pd
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    """
    PyTorch Dataset for NCF with on-the-fly negative sampling.
    Each __getitem__ returns (user, item, label), where label=1 for positives, 0 for negatives.
    """
    def __init__(self, ratings, num_items, num_negatives=3):
        """
        ratings: pd.DataFrame with columns ['userId', 'itemId', 'rating']
        num_items: total number of items in the dataset
        num_negatives: how many negatives to sample per positive
        """
        self.ratings = ratings
        self.num_items = num_items
        self.num_negatives = num_negatives

        # Build user -> set(items) mapping for fast negative sampling
        self.user_item_set = ratings.groupby('userId')['itemId'].apply(set).to_dict()
        self.users = ratings['userId'].values
        self.items = ratings['itemId'].values

    def __len__(self):
        # Each positive will be repeated (1 + num_negatives) times
        return len(self.ratings) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_negatives)
        neg_idx = idx % (1 + self.num_negatives)

        user = self.users[pos_idx]

        if neg_idx == 0:
            # Positive sample
            item = self.items[pos_idx]
            label = 1.0
        else:
            # Negative sample: draw until we find an item user hasn't interacted with
            while True:
                neg_item = random.randint(0, self.num_items - 1)
                if neg_item not in self.user_item_set[user]:
                    item = neg_item
                    label = 0.0
                    break

        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        # Filter out users with only one interaction
        user_counts = ratings['userId'].value_counts()
        multi_inter_users = user_counts[user_counts > 1].index
        ratings = ratings[ratings['userId'].isin(multi_inter_users)]

        self.ratings = ratings
        self.num_users = ratings['userId'].max() + 1
        self.num_items = ratings['itemId'].max() + 1
        self.preprocess_ratings = self._binarize(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _binarize(self, ratings):
        ratings = ratings.copy()
        ratings.loc[ratings['rating'] >= 0, 'rating'] = 1.0
        return ratings

    def _split_loo(self, ratings):
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        train_dataset = NCFDataset(self.train_ratings, self.num_items, num_negatives)
        return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def eval_batch_generator(self, evaluate_data, batch_size):
        n = len(evaluate_data['userId'])
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield {k: v[start:end] for k, v in evaluate_data.items()}

    def get_evaluation_with_negatives(self, num_negatives=99):
        """
        For each user in the test set, sample num_negatives items the user has not interacted with.
        Returns a dict with keys:
        - userId: [user1, user2, ...]
        - itemId: [pos_item1, pos_item2, ...]
        - negativeItemIds: [[neg1, neg2, ...], [neg1, neg2, ...], ...]  # list of lists
        """
        user_item_set = self.ratings.groupby('userId')['itemId'].apply(set).to_dict()
        all_items = set(range(self.num_items))
        user_ids = []
        pos_items = []
        neg_items_list = []

        for idx, row in self.test_ratings.iterrows():
            user = row['userId']
            pos_item = row['itemId']
            interacted_items = user_item_set[user]
            negative_items = list(all_items - interacted_items)
            sampled_negatives = np.random.choice(
                negative_items, size=num_negatives, replace=len(negative_items) < num_negatives
            ).tolist()
            user_ids.append(user)
            pos_items.append(pos_item)
            neg_items_list.append(sampled_negatives)

        return {
            'userId': user_ids,
            'itemId': pos_items,
            'negativeItemIds': neg_items_list
        }

    @property
    def evaluate_data(self):
        # You can adjust num_negatives as needed (default 99)
        return self.get_evaluation_with_negatives(num_negatives=99)