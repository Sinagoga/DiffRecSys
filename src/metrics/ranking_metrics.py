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

    def __call__(self, preds: Tensor, labels: Tensor, **batch):
        pos_index = calculate_pos_index(preds, labels, self.k)
        return recall_at_k(pos_index, self.k).mean()

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

    def __call__(self, preds: Tensor, labels: Tensor, **batch):
        pos_index = calculate_pos_index(preds, labels, self.k)
        return ndcg_at_k(pos_index, self.k, self.use_only_first_hit).mean()
