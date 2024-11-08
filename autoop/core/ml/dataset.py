import io

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """Dataset class to store the data frame."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the dataset object."""
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
            data: pd.DataFrame,
            name: str,
            asset_path: str,
            version: str = "1.0.0"
    ) -> 'Dataset':
        """
        Create a dataset from a data frame.
        Args:
            data: data frame to be saved.
            name: name of the dataset.
            asset_path: path to save the dataset.
            version: version of the dataset.

        Returns:
            The dataset object.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read_as_data_frame(self) -> pd.DataFrame:
        """
        Read the data frame from the dataset.
        Returns: The data frame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save the data frame as a csv file.
        Args:
            data: - data frame to be saved.

        Returns: The bytes of the saved data frame.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
