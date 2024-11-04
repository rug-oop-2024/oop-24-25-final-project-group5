
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """Feature class that describes the type of the feature."""
    name: str = Field()
    type: Literal["categorical", "numerical"] = Field()

    def __str__(self):
        return f"Feature with column name {self.name} and of type {self.type}"
