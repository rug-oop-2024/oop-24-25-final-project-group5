from abc import ABC, abstractmethod
import pandas as pd
from pydantic import BaseModel, Field


class Slicer(ABC, BaseModel):
    """
    Abstract class for slicers. This class defines
    the interface for different slicer implementations.
    """
    @abstractmethod
    def should_include(self, row: pd.Series) -> bool:
        """
        Check if a row should be included in the
        sliced data

        Args:
            row: Row to check

        Returns: True if the row should be included, False otherwise
        """
        pass

    def slice(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Slice data based on the given slicer

        Args:
            data: Data to slice

        Returns: Sliced data
        """
        result = data.copy()
        for i, row in data.iterrows():
            if not self.should_include(row):
                result = result.drop(i)
        return result


class NumericRangeSlicer(Slicer, BaseModel):
    """
    Slicer for numerical data. This will slice
    data based on a given column and a range
    of values to include.
    """
    column: str = Field()
    min: float = Field()
    max: float = Field()

    def should_include(self, row: pd.Series) -> bool:
        """
        Check if a row should be included in the
        sliced data

        Args:
            row: Row to check

        Returns: True if the row should be included, False otherwise

        """
        return self.min <= row[self.column] <= self.max


class CategoricalSlicer(Slicer, BaseModel):
    """
    Slicer for categorical data. This will slice
    data based on a given column and a list of
    categories to include.
    """
    column: str = Field()
    categories: list[str] = Field()

    def should_include(self, row: pd.Series) -> bool:
        """
        Check if a row should be included in the
        sliced data
        Args:
            row: Row to check

        Returns: True if the row should be included, False otherwise
        """
        if self.categories is None or len(self.categories) == 0:
            return True

        return row[self.column] in self.categories
