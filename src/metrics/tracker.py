import pandas as pd


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self):
        self._data = {}
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.keys():
            self._data[col] = {
                "total": 0,
                "counts": 0,
            }

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        if key not in self._data:
            self._data[key] = {
                "total": 0,
                "counts": 0,
            }
        self._data[key]["total"] += value * n
        self._data[key]["counts"] += n

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data[key]["total"] / self._data[key]["counts"]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return {
            k: v["total"] / v["counts"]
            for k, v in self._data.items()
        }

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.keys()
