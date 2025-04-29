import math

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self.reset()

    def reset(self):
        self._num_users = 0
        self._num_hits = 0
        self._ndcg_sum = 0.0

    def update(self, test_score, neg_scores):
        # test_score: float, the predicted score for the positive item
        # neg_scores: list of floats, predicted scores for negative items
        all_scores = neg_scores + [test_score]
        rank = 1 + sum([s > test_score for s in neg_scores])  # 1-based rank
        if rank <= self._top_k:
            self._num_hits += 1
            self._ndcg_sum += math.log(2) / math.log(1 + rank)
        self._num_users += 1

    def cal_hit_ratio(self):
        return self._num_hits / self._num_users if self._num_users > 0 else 0.0

    def cal_ndcg(self):
        return self._ndcg_sum / self._num_users if self._num_users > 0 else 0.0