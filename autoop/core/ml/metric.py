from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> any:
        """Call method for any Metric object

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated metric
        """
        return self.evaluate(ground_truth, prediction)

    @abstractmethod
    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> any:
        """Abstract evaluate method to be implemented
        separately in any class that inherits from
        the abstract class Metric.

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated metric
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Abstract method for printing the metric's name.

        Returns:
            str: formatted metric name.
        """
        pass


# Regression Metrics
class MeanSquaredError(Metric):
    """Mean Squared Error metric."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Evaluate method for the mean squared error metric.

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated mean squared error
        """
        return np.mean((ground_truth - prediction) ** 2)

    def __str__(self) -> str:
        """Method for printing the metric's name.

        Returns:
            str: formatted metric name.
        """
        return "Mean Squared Error"


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Evaluate method for the mean absoulte error metric.

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated mean absolute error
        """
        return np.mean(np.abs(ground_truth - prediction))

    def __str__(self) -> str:
        """Method for printing the metric's name.

        Returns:
            str: formatted metric name.
        """
        return "Mean Absolute Error"


class R2Score(Metric):
    """R-squared (R^2) score."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Evaluate method for the R-squared score metric.

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated R-squared score
        """
        regression_soq = np.sum((ground_truth - prediction) ** 2)
        total_soq = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        return 1 - (regression_soq / total_soq)

    def __str__(self) -> str:
        """Method for printing the metric's name.

        Returns:
            str: formatted metric name.
        """
        return "R2 Score"


# Classification Metrics
class AccuracyMetric(Metric):
    """Calculates accuracy score."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Evaluate method for the accuracy score metric.

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated accuracy score
        """
        return np.mean(ground_truth == prediction)

    def __str__(self) -> str:
        """Method for printing the metric's name.

        Returns:
            str: formatted metric name.
        """
        return "Accuracy"


class PrecisionMetric(Metric):
    """Calculates precision score based on average."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Evaluate method for the precision score metric.

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated precision score
        """
        true_positive = np.sum((ground_truth == 1) & (prediction == 1))
        false_positive = np.sum((ground_truth == 0) & (prediction == 1))
        return true_positive / (true_positive + false_positive)

    def __str__(self) -> str:
        """Method for printing the metric's name.

        Returns:
            str: formatted metric name.
        """
        return "Precision"


class RecallMetric(Metric):
    """Calculates metric score based on average."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Evaluate method for the recall score metric.

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated recall score
        """
        true_positive = np.sum((ground_truth == 1) & (prediction == 1))
        false_negative = np.sum((ground_truth == 1) & (prediction == 0))
        return true_positive / (true_positive + false_negative)

    def __str__(self) -> str:
        """Method for printing the metric's name.

        Returns:
            str: formatted metric name.
        """
        return "Recall"


class F1Score(Metric):
    """Calculates metric score based on average."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Evaluate method for the F1 score metric.

        Arguments:
            ground_truth (np.ndarray): list of ground truth values
            prediction (np.ndarray): list of prediction values

        Returns:
            float: calculated F1 score
        """
        precision = PrecisionMetric()(ground_truth, prediction)
        recall = RecallMetric()(ground_truth, prediction)
        return (2 * (precision * recall)) / (precision + recall)

    def __str__(self) -> str:
        """Method for printing the metric's name.

        Returns:
            str: formatted metric name.
        """
        return "F1 Score"


def get_metric(name: str) -> Metric:
    """
    Returns the metric object based
    on the name provided.
    Args:
        name: name of the metric

    Returns: metric object
    """
    return METRICS_MAP[name]


METRICS_MAP = {
    "mean_squared_error": MeanSquaredError(),
    "mean_absolute_error": MeanAbsoluteError(),
    "r2_score": R2Score(),
    "accuracy": AccuracyMetric(),
    "precision": PrecisionMetric(),
    "recall": RecallMetric(),
    "f1_score": F1Score()
}


METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]
