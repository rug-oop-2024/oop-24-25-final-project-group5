from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from sklearn import metrics

METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",

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


class PrecisionMetric(Metric):
    """Precision metric.
    """
    def evaluate(self, ground_truth: list[float], prediction: list[float]):
        ground_truth = np.array(ground_truth)
        prediction = np.array(prediction)

        true_positive = np.sum((ground_truth == 1) & (prediction == 1))
        false_positive = np.sum((ground_truth == 0) & (prediction == 1))
        return true_positive / (true_positive + false_positive)
 

class RecallMetric(Metric):
    """Recall metric.
    """
    def evaluate(self, ground_truth: list[float], prediction: list[float]):
        ground_truth = np.array(ground_truth)
        prediction = np.array(prediction)

        true_positive = np.sum((ground_truth == 1) & (prediction == 1))
        false_negative = np.sum((ground_truth == 1) & (prediction == 0))
        return true_positive / (true_positive + false_negative)


METRICS_MAP = {
    "mean_squared_error": MeanSquaredError(),
    "accuracy": AccuracyMetric(),
    "precision": PrecisionMetric(),
    "recall": RecallMetric
}

true_values = [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
predictions = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]

print(RecallMetric().evaluate(true_values, predictions))
print(metrics.recall_score(true_values, predictions))