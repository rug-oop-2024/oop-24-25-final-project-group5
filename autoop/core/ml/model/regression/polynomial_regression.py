from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression(Model):
    """
    Polynomial regression model.
    """
    def __init__(self, degree: int = 2) -> None:
        """Initializes the model and sets the hyperparameters
        based on type of model. Hyperparameters are listed as arguments.

        Arguments:
            degree (int): degree of polynomial features, default is 2.
        """
        super().__init__()
        # set degree as hyperparameter
        if degree <= 0:
            raise ValueError("Cannot have a <=0 degree for the polynomial.")

        self.hyperparameters = {"degree": degree}
        self.hyperparameter_descriptions = {
            "degree": "Degree of polynomial features"
        }
        self.type = "regression"

        self._linear_model = LinearRegression()
        self._poly_features = PolynomialFeatures(
            degree=degree
        )

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
        # transform to polynomial features
        X_poly = self._poly_features.fit_transform(observations)
        # fit linear regression model
        self._linear_model.fit(X_poly, ground_truth)
        # store parameters
        self._parameters = {"coef_": self._linear_model.coef_,
                            "intercept_": self._linear_model.intercept_
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
        # transform observations to polynomial features
        X_poly = self._poly_features.transform(observations)
        # use fitted model to predict
        return self._linear_model.predict(X_poly)
