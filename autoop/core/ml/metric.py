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
    def evaluate(self, ground_truth: list[Any], prediction: list[Any]):
        pass

class MeanSquaredError(Metric):
    """Mean Squared Error metric.
    """

    def evaluate(self, ground_truth: list[float], prediction: list[float]):
        return np.mean((np.array(ground_truth) - np.array(prediction)) ** 2)


class AccuracyMetric(Metric):
    """Accuracy metric.
    """

    def evaluate(self, ground_truth: list[float], prediction: list[float]):
        return np.mean(np.array(ground_truth) == np.array(prediction))



METRICS_MAP = {
    "mean_squared_error": MeanSquaredError(),
    "accuracy": AccuracyMetric(),
}