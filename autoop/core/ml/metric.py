from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    return METRICS_MAP[name]

class Metric(ABC):
    """Base class for all metrics.
    """

    @abstractmethod
    def __call__(self, ground_truth: list[Any], prediction: list[Any]):
        pass

class MeanSquaredErrorMetric(Metric):
    """Mean Squared Error metric.
    """

    def __call__(self, ground_truth: list[float], prediction: list[float]):
        return np.mean((np.array(ground_truth) - np.array(prediction)) ** 2)


class AccuracyMetric(Metric):
    """Accuracy metric.
    """

    def __call__(self, ground_truth: list[float], prediction: list[float]):
        return np.mean(np.array(ground_truth) == np.array(prediction))



METRICS_MAP = {
    "mean_squared_error": MeanSquaredErrorMetric(),
    "accuracy": AccuracyMetric(),
}