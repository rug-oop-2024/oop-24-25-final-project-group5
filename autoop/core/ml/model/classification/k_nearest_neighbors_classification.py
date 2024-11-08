from autoop.core.ml.model.model import Model
import numpy as np
from collections import Counter


class KNearestNeighborsClassification(Model):
    """Class that implements the k-nearest neighbor algorithm."""

    def __init__(self, k: int = 3) -> None:
        """Initializes the model and sets the hyperparameters
        based on the type of model. Hyperparameters are listed as arguments.

        Arguments:
            k (int): number of nearest neighbors to be considered
                     during predictions, default is 3.
        """
        if k < 1:
            raise ValueError("k must be greater than 1.")

        super().__init__()
        self.type = "classification"
        self.hyperparameters = {"k": k}

        self.hyperparameter_descriptions = {
            "k": "Number of neighbors to consider for classification"
        }

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Method that stores the models parameters in a dictionary.

        Arguments:
            observations (np.ndarray): row(s) of a dataset
                                       used for training.
            ground_truth (np.ndarray): value of response for
                                       given observations.
        """
        self.parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Method that returns predictions based on
        a set of observations.

        Arguments:
            observations (np.ndarray): row(s) of a dataset
                                       used for predicting.
        Returns:
            predicted behavior of observation as np.ndarray.
        """
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> np.ndarray:
        """Submethod of predict that predicts
        a single row of observations.

        Arguments:
            observations (np.ndarray): row of the dataset
                                       used for predicting.
        Returns:
            most common label of observations as np.ndarray.
        """
        # Access k
        k = self.hyperparameters["k"]
        # Calculate distance between observation and every other point
        distances = np.linalg.norm(
            self.parameters["observations"] - observation,
            axis=1
        )
        # Sort the array of the distances and take first k
        k_indices = np.argsort(distances)[:k]
        # Check the label aka ground truth of those points
        k_nearest_labels = [self.parameters["ground_truth"][i]
                            for i in k_indices]
        # Convert k_nearest_labels list of np.arrays to tuples,
        # so that counter works
        tuples_nearest_labels = [tuple(array) for array in k_nearest_labels]
        # Take the most common label and return it to the caller
        most_common_array = Counter(tuples_nearest_labels).most_common()[0][0]
        return np.array(most_common_array)
