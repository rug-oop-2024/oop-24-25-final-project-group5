from abc import ABC, abstractmethod
from typing import Any
import numpy as np


METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]


def get_metric(name: str):
    return METRICS_MAP[name]


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(self, ground_truth: list[Any],
                 prediction: list[Any]) -> any:
        """Call method for any Metric object

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values

        Returns:
            float: calculated metric
        """
        return self.evaluate(ground_truth, prediction)

    @abstractmethod
    def evaluate(self, ground_truth: list[Any],
                 prediction: list[Any]) -> any:
        """Abstract evaluate method to be implemented
        separately in any class that inherits from
        the abstract class Metric.

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values

        Returns:
            float: calculated metric
        """
        pass


# Regression Metrics
class MeanSquaredError(Metric):
    """Mean Squared Error metric."""

    def evaluate(self, ground_truth: list[float],
                 prediction: list[float]) -> float:
        """Evaluate method for the mean squared error metric.

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values

        Returns:
            float: calculated mean squared error
        """
        return np.mean((np.array(ground_truth) - np.array(prediction)) ** 2)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric."""

    def evaluate(self, ground_truth: list[float],
                 prediction: list[float]) -> float:
        """Evaluate method for the mean absoulte error metric.

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values

        Returns:
            float: calculated mean absolute error
        """
        return np.mean(np.abs(np.array(ground_truth) - np.array(prediction)))


class R2Score(Metric):
    """R-squared (R^2) score."""

    def evaluate(self, ground_truth: list[float],
                 prediction: list[float]) -> float:
        """Evaluate method for the R-squared score metric.

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values

        Returns:
            float: calculated R-squared score
        """
        ground_truth = np.array(ground_truth)
        prediction = np.array(prediction)

        regression_soq = np.sum((ground_truth - prediction) ** 2)
        total_soq = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        return 1 - (regression_soq/total_soq)


# Classification Metrics
class AccuracyMetric(Metric):
    """Calculates accuracy score."""

    def evaluate(self, ground_truth: list[float],
                 prediction: list[float]) -> float:
        """Evaluate method for the accuracy score metric.

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values

        Returns:
            float: calculated accuracy score
        """
        return np.mean(np.array(ground_truth) == np.array(prediction))


class PrecisionMetric(Metric):
    """Calculates precision score based on average."""

    def evaluate(self, ground_truth: list[float],
                 prediction: list[float],
                 average: str = 'micro') -> float:
        """Evaluate method for the precision score metric.
        Average is used so that this metric can be used on
        generic multi-class classification tasks.

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values
            average (str): string of average method (micro or macro)

        Returns:
            float: calculated precision score
        """
        ground_truth = np.array(ground_truth)
        prediction = np.array(prediction)

        uniques = np.unique(ground_truth)
        precision_per_class = []
        for cls in uniques:
            true_positive = np.sum((ground_truth == cls) & (prediction == cls))
            false_positive = np.sum((
                ground_truth != cls) & (prediction == cls)
            )
            precision_per_class.append(
                (true_positive / (true_positive + false_positive))
            )

        if average == "micro":
            return np.sum((ground_truth == prediction)) / len(ground_truth)
        elif average == "macro":
            return np.mean(precision_per_class)


class RecallMetric(Metric):
    """Calculates metric score based on average."""

    def evaluate(self, ground_truth: list[float],
                 prediction: list[float],
                 average: str = 'micro') -> float:
        """Evaluate method for the recall score metric.
        Average is used so that this metric can be used on
        generic multi-class classification tasks.

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values
            average (str): string of average method (micro or macro)

        Returns:
            float: calculated recall score
        """
        ground_truth = np.array(ground_truth)
        prediction = np.array(prediction)

        uniques = np.unique(ground_truth)
        recall_per_class = []
        for cls in uniques:
            true_positive = np.sum((ground_truth == cls) & (prediction == cls))
            false_negative = np.sum(
                (ground_truth == cls) & (prediction != cls)
            )
            recall_per_class.append(
                (true_positive / (true_positive + false_negative))
            )

        if average == "micro":
            return np.sum((ground_truth == prediction)) / len(ground_truth)
        elif average == "macro":
            return np.mean(recall_per_class)


class F1Score(Metric):
    """Calculates metric score based on average."""

    def evaluate(self, ground_truth: list[float],
                 prediction: list[float],
                 average: str = 'micro') -> float:
        """Evaluate method for the F1 score metric.
        Average is used so that this metric can be used on
        generic multi-class classification tasks.

        Arguments:
            ground_truth (list[Any]): list of ground truth values
            prediction (list[Any]): list of prediction values
            average (str): string of average method (micro or macro)

        Returns:
            float: calculated F1 score
        """
        ground_truth = np.array(ground_truth)
        prediction = np.array(prediction)
        uniques = np.unique(ground_truth)
        f1_per_class = []
        if average == 'macro':
            for cls in uniques:
                tp = np.sum((ground_truth == cls) & (prediction == cls))
                fp = np.sum((ground_truth != cls) & (prediction == cls))
                fn = np.sum((ground_truth == cls) & (prediction != cls))
                precision = tp / (tp + fp)
                recall = tp / (fp + fn)
                f1 = (2 * (precision * recall)) / (precision + recall)
                f1_per_class.append(f1)
            return np.mean(f1_per_class)
        precision = PrecisionMetric().evaluate(
            ground_truth, prediction, average
        )
        recall = RecallMetric().evaluate(ground_truth, prediction, average)
        return (2 * (precision * recall)) / (precision + recall)


METRICS_MAP = {
    "mean_squared_error": MeanSquaredError(),
    "mean_absolute_error": MeanAbsoluteError(),
    "r2_score": R2Score(),
    "accuracy": AccuracyMetric(),
    "precision": PrecisionMetric(),
    "recall": RecallMetric(),
    "f1_score": F1Score()
}
