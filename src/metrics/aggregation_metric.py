from typing import Optional

import torch.nn as nn

from src.metrics.base_metric import BaseMetric


class AggregationMetric(BaseMetric):
    def __init__(
            self,
            base_metrics: list[BaseMetric],
            weights: Optional[list[float]] = None,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.base_metrics = nn.ModuleList(base_metrics)
        if weights is not None:
            assert len(weights) == len(base_metrics), "Weights length must match base metrics length"
            self.weights = weights
        else:
            self.weights = [1 / len(base_metrics)] * len(base_metrics)

    def update(self, **batch):
        for metric in self.base_metrics:
            metric.update(**batch)
            
    def compute(self):
        return sum(
            metric.compute() * weight
            for metric, weight in zip(self.base_metrics, self.weights)
        )
