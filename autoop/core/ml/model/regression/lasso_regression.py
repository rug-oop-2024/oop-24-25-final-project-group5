from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Lasso


class LassoRegression(Model):
    """
    Lasso regression model.
    """
    def __init__(self,
                 alpha: int = 1.0,
                 max_iter: int = 1000,
                 tol: int = 0.0001,
                 selection: str = 'cyclic') -> None:
        """Initializes the model and sets the hyperparameters
        based on type of model. Hyperparameters are listed as arguments.

        Arguments:
            alpha (int): constant that multiplies the model's
                         L1 term, default is 1.0.
            max_iter (int): maximum number of iterations, default is 1000.
            tol (int): tolerance of the optimization, default is 0.0001.
            selection (str): changes how coefficients are looped over,
                             either 'cyclic' or 'random', default is 'cyclic'.
        """
        super().__init__()
        self.type = "regression"

        if alpha == 0:
            raise ValueError("Cannot have a 0 alpha in a Lasso Regression")

        # set hyperparameters
        self.hyperparameters = {
            "alpha": alpha,
            "max_iter": max_iter,
            "tol": tol,
            "selection": selection
        }

        self.hyperparameter_descriptions = {
            "alpha": "Constant that multiplies the L1 term. Defaults to 1.0.",
            "max_iter": "The maximum number of iterations. Defaults to 1000.",
            "tol": "The tolerance for the optimization. Defaults to 0.0001.",
            "selection": "If set to 'random', a random coefficient is "
                         "updated every iteration rather than looping over "
                         "features sequentially. Defaults to 'cyclic'."
        }

        self._lasso_model = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            selection=selection
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
        self._lasso_model.fit(observations, ground_truth)
        self.parameters = {
            "coeff_": self._lasso_model.coef_,
            "intercept_": self._lasso_model.intercept_
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
        return self._lasso_model.predict(observations)
