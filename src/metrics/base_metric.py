from abc import abstractmethod

from torchmetrics import Metric


class BaseMetric(Metric):
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        super().__init__(**kwargs)
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def update(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        """
        raise NotImplementedError()
