from abc import ABC, abstractmethod
import pandas as pd


class Slicer(ABC):
    """
    Abstract class for slicers. This class defines
    the interface for different slicer implementations.
    """

    @abstractmethod
    def should_include(self, row: pd.Series) -> bool:
        """Check if a row should be included in the
        sliced data

        Args:
            row (pd.Series): Row to check.

        Returns: True if the row should be included, False otherwise
        """
        pass

    def slice(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Slice data based on the given slicer.

        Args:
            data (pd.DataFrame): Data to slice.

        Returns: Sliced data.
        """
        result = data.copy()
        for i, row in data.iterrows():
            if not self.should_include(row):
                result = result.drop(i)
        return result


class NumericRangeSlicer(Slicer):
    """
    Slicer for numerical data. This will slice
    data based on a given column and a range
    of values to include.
    """

    def __init__(self, column: str, min: float, max: float) -> None:
        """Initializes the numeric slicer.

        Arguments:
            column (str): column to be sliced.
            min (float): minimum range.
            max (float): maximum range.
        """
        self.column = column
        self.min = min
        self.max = max

    def should_include(self, row: pd.Series) -> bool:
        """Check if a row should be included in the
        sliced data

        Args:
            row: Row to check

        Returns: True if the row should be included, False otherwise

        """
        return self.min <= row[self.column] <= self.max


class CategoricalSlicer(Slicer):
    """
    Slicer for categorical data. This will slice
    data based on a given column and a list of
    categories to include.
    """

    def __init__(self, column: str, categories: list[str]) -> None:
        """Initializes the numeric slicer.

        Arguments:
            column (str): column to be sliced.
            categories (list[str]): categories to be included
        """
        self.column = column
        self.categories = categories

    def should_include(self, row: pd.Series) -> bool:
        """Check if a row should be included in the
        sliced data

        Arguments:
            row: Row to check

        Returns: True if the row should be included, False otherwise
        """
        if self.categories is None or len(self.categories) == 0:
            return True

        return row[self.column] in self.categories
