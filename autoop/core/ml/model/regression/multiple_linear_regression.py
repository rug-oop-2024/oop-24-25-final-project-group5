from autoop.core.ml.model.model import Model

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(Model):
    def __init__(self):
        # Inherit from Model
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        # Store observations and ground_truth in parameters
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

        # Add a column of 1 for the intercept
        observations_bias = np.insert(observations, 0, 1, axis=1)

        # w= (X^T * X)^-1 * X^T * y
        X_T_X = np.dot(observations_bias.T, observations_bias)
        X_T_X_inv = np.linalg.pinv(X_T_X)
        X_T_y = np.dot(observations_bias.T, ground_truth)
        w = np.dot(X_T_X_inv, X_T_y)

        # Store the parameters
        self._parameters["parameters"] = w

    def predict(self, observations: np.ndarray) -> np.ndarray:
        # Check if the model has been trained
        if self._parameters["parameters"] is None:
            print("The model is not trained yet.")

        # Add a column of 1 for the intercept
        observations_bias = np.insert(observations, 0, 1, axis=1)

        # Read the parameters
        beta = self._parameters["parameters"]

        # y = X * w
        predictions = np.dot(observations_bias, beta)

        return predictions

    #Allow users to read parameters without modifying them
    @property
    def parameters(self):
        return self._parameters
