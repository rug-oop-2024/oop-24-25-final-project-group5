from autoop.core.ml.model.model import Model

import numpy as np

from collections import Counter
from pydantic import field_validator

# Inherit from Model
class KNearestNeighbors(Model):
    # Add attribute k (public), initialize observations and ground_truth as None
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.observations = None
        self.ground_truth = None

    # Check that k >= 0
    @field_validator("k")
    def k_greater_than_zero(cls, value):
        if value <= 0:
            raise ValueError("k must be greater than 0.")

    # Store observations and ground_truth in parameters
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self.observations = observations
        self.ground_truth = ground_truth
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray):
        predictions = [self._predict_single(x) for x in observations]
        return predictions

    # Predict a single point
    def _predict_single(self, observation: np.ndarray):
        
        # Calculate distance between observation and every other point
        distances = np.linalg.norm(self._parameters["observations"]
                                   - observation, axis=1)
        # Sort the array of the distances and take first k
        k_indices = np.argsort(distances)[:self.k]
        # Check the label aka ground truth of those points
        k_nearest_labels = [self._parameters["ground_truth"][i]
                            for i in k_indices]
        # Take the most common label and return it to the caller
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

    # Allow user to read parameters without modifying them
    @property
    def parameters(self):
        return self._parameters
