from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression(Model):
    def __init__(self, degree=2) -> None:
        super().__init__()
        # set degree as hyperparameter
        self.hyperparameters = {"degree": degree}
        self.hyperparameter_descriptions = {
            "degree": "Degree of polynomial features"
        }
        self.type = "regression"

        # initialize linear regression model
        self._linear_model = LinearRegression()
        # polynomial features
        self._poly_features = PolynomialFeatures(
            degree=self.hyperparameters["degree"]
        )

    # fit: transform to polynomial features and train linear regression model
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        # transform to polynomial features
        X_poly = self._poly_features.fit_transform(observations)
        # fit linear regression model
        self._linear_model.fit(X_poly, ground_truth)
        # store parameters
        self.parameters = {"coef_": self._linear_model.coef_}
        self.parameters = {"intercept_": self._linear_model.intercept_}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        # transform observations to polynomial features
        X_poly = self._poly_features.transform(observations)
        # use fitted model to predict
        return self._linear_model.predict(X_poly)
