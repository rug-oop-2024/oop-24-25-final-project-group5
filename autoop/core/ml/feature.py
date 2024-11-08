from typing import Literal


class Feature():
    """Feature class that describes the name and type of the feature."""

    def __init__(self,
                 name: str,
                 type: Literal["categorical", "numerical"]) -> None:
        self.name = name
        self.type = type

    def __str__(self) -> str:
        """Returns a formatted string representing the feature."""
        return f"Feature with name {self.name} and type {self.type}"
