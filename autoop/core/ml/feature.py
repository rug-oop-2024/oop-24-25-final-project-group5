from typing import Literal


class Feature():
    """Feature class that describes the name and type of the feature."""

    def __init__(self,
                 name: str,
                 type: Literal["categorical", "numerical"]) -> None:
        """Initializes a feature

        Arguments:
            name (str): name of the feature/column
            type (Literal["categorical", "numerical"]):
                type of feature, either categorical or numerical
        """
        self.name = name
        self.type = type

    def __eq__(self, other: 'Feature') -> bool:
        if not isinstance(other, Feature):
            return False
        return self.name == other.name and self.type == other.type

    def __str__(self) -> str:
        """Returns a formatted string representing the feature."""
        return f"Feature with name {self.name} and type {self.type}"
