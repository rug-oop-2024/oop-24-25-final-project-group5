from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Lasso


# Inherit from Model
class LassoWrapper(Model):
    def __init__(self, alpha=1.0, max_iter=1000,
                 tol=0.0001, selection='cyclic'):
        super().__init__()
        # set hyperparameters
        self.hyperparameters = {
            "alpha": alpha,
            "max_iter": max_iter,
            "tol": tol,
            "selection": selection
        }
        # initialize Lasso model with hyperparameters
        self._lasso_model = Lasso(
            alpha=self.hyperparameters["alpha"],
            max_iter=self.hyperparameters["max_iter"],
            tol=self.hyperparmeters["tol"],
            selection=self.hyperparameters["selection"]
        )

    # Fit data, store coefficient and intercept in parameters
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self._lasso_model.fit(observations, ground_truth)
        # Store learned parameters
        self.parameters = {
            "coeff_": self._lasso_model.coef_,
            "intercept_": self._lasso_model.intercept_
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._lasso_model.predict(observations)
