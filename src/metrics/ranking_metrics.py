import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calculate_pos_index, recall_at_k, ndcg_at_k


class RecallAtK(BaseMetric):
    def __init__(
            self,
            k: int,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.k = k
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, labels: Tensor, **batch):
        pos_index = calculate_pos_index(preds, labels, self.k)
        batch_sum = recall_at_k(pos_index, self.k).sum()
        self.sum += batch_sum
        self.count += pos_index.shape[0]

    def compute(self):
        return self.sum / self.count

class NDCGAtK(BaseMetric):
    def __init__(
            self,
            k: int,
            use_only_first_hit: bool = True,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.k = k
        self.use_only_first_hit = use_only_first_hit
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, labels: Tensor, **batch):
        pos_index = calculate_pos_index(preds, labels, self.k)
        batch_sum = ndcg_at_k(pos_index, self.k, self.use_only_first_hit).sum()
        self.sum += batch_sum
        self.count += pos_index.shape[0]
        
    def compute(self):
        return self.sum / self.count

