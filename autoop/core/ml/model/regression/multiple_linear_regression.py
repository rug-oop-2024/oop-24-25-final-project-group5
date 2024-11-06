from autoop.core.ml.model.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    def __init__(self) -> None:
        """Initializes the model."""
        super().__init__()
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Method that fits the model based on
        observations and their ground truth and stores
        the model's parameters in a dictionary.

        Arguments:
            observations (np.ndarray): row(s) of a dataset
                                       used for training.
            ground_truth (np.ndarray): value of response for
                                        given observations.
        """
        # Add a column of 1 for the intercept
        observations_bias = np.insert(observations, 0, 1, axis=1)

        # w = (X^T * X)^-1 * X^T * y
        X_T_X = np.dot(observations_bias.T, observations_bias)
        X_T_X_inv = np.linalg.pinv(X_T_X)
        X_T_y = np.dot(observations_bias.T, ground_truth)
        w = np.dot(X_T_X_inv, X_T_y)

        self.parameters = {"parameters": w}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Method that returns predictions based on
        a set of observations.

        Arguments:
            observations (np.ndarray): row(s) of a dataset
                                       used for predicting.
        Returns:
            predicted behavior of observation as np.ndarray.
        """
        # Check if the model has been trained
        if not self.parameters["parameters"]:
            print("The model is not trained yet.")

        # Add a column of 1 for the intercept
        observations_bias = np.insert(observations, 0, 1, axis=1)

        # Read the parameters
        beta = self.parameters["parameters"]

        # y = X * w
        predictions = np.dot(observations_bias, beta)

        return predictions
