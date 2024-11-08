from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """Feature class that describes the name and type of the feature."""

    name: str = Field()
    type: Literal["categorical", "numerical"] = Field()

    def __str__(self) -> str:
        """Returns a formatted string representing the feature."""
        return f"Feature with name {self.name} and type {self.type}"
