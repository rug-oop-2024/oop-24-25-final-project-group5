from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model(ABC):
    _parameters: dict = {}
    _hyperparameters: dict = {}

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, new_params: dict) -> None:
        self._parameters.update(new_params)

    @property
    def hyperparameters(self) -> dict:
        return deepcopy(self._hyperparameters)

    @hyperparameters.setter
    def hyperparameters(self, new_hyperparams: dict) -> None:
        self._hyperparameters.update(new_hyperparams)
