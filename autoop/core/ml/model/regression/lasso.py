from autoop.core.ml.model.model import Model

import numpy as np
from sklearn.linear_model import Lasso

# Inherit from Model
class LassoWrapper(Model):
    # Add Lasso model and required parameters to inherited __init__
    def __init__(self):
        super().__init__()
        self._lasso_model = Lasso()
        self._parameters = {
            "alpha": self._lasso_model.alpha,
            "max_iter": self._lasso_model.max_iter,
            "tol": self._lasso_model.tol,
            "selection": self._lasso_model.selection
        }

    # Fit data, store coefficient and intercept in parameters
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self._lasso_model.fit(observations, ground_truth)
        self._parameters["coef_"] = self._lasso_model.coef_
        self._parameters["intercept_"] = self._lasso_model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._lasso_model.predict(observations)

    # Allow user to read parameters without modifying them
    @property
    def parameters(self):
        return self._parameters
