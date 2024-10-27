from abc import ABC, abstractmethod
import pandas as pd
from pydantic import BaseModel, Field

class Slicer(ABC, BaseModel):
    @abstractmethod
    def should_include(self, row: pd.Series) -> bool:
        pass

    def slice(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for i, row in data.iterrows():
            if not self.should_include(row):
                result = result.drop(i)
        return result

class NumericRangeSlicer(Slicer, BaseModel):
    column: str = Field()
    min: float = Field()
    max: float = Field()

    def should_include(self, row: pd.Series) -> bool:
        return self.min <= row[self.column] <= self.max

class CategoricalSlicer(Slicer, BaseModel):
    column: str = Field()
    categories: list[str] = Field()

    def should_include(self, row: pd.Series) -> bool:
        if self.categories is None or len(self.categories) == 0:
            return True

        return row[self.column] in self.categories