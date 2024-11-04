from autoop.core.ml.model.model import Model
import numpy as np
from collections import Counter
from pydantic import field_validator


# Inherit from Model
class KNearestNeighborsClassification(Model):
    def __init__(self, k=3) -> None:
        super().__init__()
        self.type = "classification"
        # store hyperparameter k
        self.hyperparameters = {"k": k}
        # set observations and ground_truth to None in parameters
        self.parameters = {"observations": None, "ground_truth": None}

        self.hyperparameter_descriptions = {
            "k": "Number of neighbors to consider for classification"
        }

    # Check that k >= 0
    @field_validator("k")
    def k_greater_than_zero(cls, value):
        if value <= 0:
            raise ValueError("k must be greater than 0.")

    # Store observations and ground_truth in parameters
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self.parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    # Predict a single point
    def _predict_single(self, observation: np.ndarray):
        # Access k
        k = self.hyperparameters["k"]
        # Calculate distance between observation and every other point
        distances = np.linalg.norm(self.parameters["observations"]
                                   - observation, axis=1)
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
