import torch
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import save_checkpoint, use_optimizer, use_cuda
from metrics import MetronAtK


class Engine(object):
    """
    Meta Engine for training & evaluating NCF model
    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.device = use_cuda(self.config['use_cuda'], self.config.get('device_id', 0))
        self.model.to(self.device)

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        self._metron.reset()
        with torch.no_grad():
            test_users = evaluate_data['userId']
            test_items = evaluate_data['itemId']
            negative_items = evaluate_data['negativeItemIds']  # shape: (num_users, num_negatives)
            bs = self.config['batch_size']

            for start_idx in range(0, len(test_users), bs):
                end_idx = min(start_idx + bs, len(test_users))
                batch_users = test_users[start_idx:end_idx]
                batch_pos_items = test_items[start_idx:end_idx]
                batch_neg_items = negative_items[start_idx:end_idx]  # shape: (batch, num_neg)

                for i in range(len(batch_users)):
                    user = batch_users.iloc[i] if hasattr(batch_users, 'iloc') else batch_users[i]
                    pos_item = batch_pos_items.iloc[i] if hasattr(batch_pos_items, 'iloc') else batch_pos_items[i]
                    negs = batch_neg_items.iloc[i] if hasattr(batch_neg_items, 'iloc') else batch_neg_items[i]

                    # Prepare tensors
                    user_tensor = torch.tensor([user] * (len(negs) + 1), dtype=torch.long).to(self.device)
                    item_tensor = torch.tensor(list(negs) + [pos_item], dtype=torch.long).to(self.device)
                    scores = self.model(user_tensor, item_tensor).cpu().numpy().tolist()
                    test_score = scores[-1]
                    neg_scores = scores[:-1]
                    self._metron.update(test_score, neg_scores)

        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evaluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)