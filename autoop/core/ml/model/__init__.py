
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import LassoRegression
from autoop.core.ml.model.regression import PolynomialRegression
from autoop.core.ml.model.classification import DecisionTreeClassification
from autoop.core.ml.model.classification import RandomForestClassification
from autoop.core.ml.model.classification import KNearestNeighborsClassification

from typing import Literal

REGRESSION_MODELS = {
    "Lasso": LassoRegression,
    "MultipleLinear": MultipleLinearRegression,
    "Polynomial": PolynomialRegression,
}

CLASSIFICATION_MODELS = {
    "DecisionTree": DecisionTreeClassification,
    "RandomForest": RandomForestClassification,
    "KNearestNeighbors": KNearestNeighborsClassification,
}

def get_models() -> dict[str, Literal["regression", "classification"]]:
    return {**{model: "regression" for model in REGRESSION_MODELS}, **{model: "classification" for model in CLASSIFICATION_MODELS}}

def get_model(model_name: str) -> type[Model]:
    if model_name in REGRESSION_MODELS:
        return REGRESSION_MODELS[model_name]
    elif model_name in CLASSIFICATION_MODELS:
        return CLASSIFICATION_MODELS[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")

