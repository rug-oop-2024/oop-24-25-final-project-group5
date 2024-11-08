from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """Artifact object used to store assets
    containing specific asset information.
    """

    name: str = Field()
    version: str = Field(default="1.0.0")
    asset_path: str = Field(default="")
    tags: list = Field(default=[])
    metadata: dict = Field(default={})
    data: bytes = Field()
    type: str = Field(default="")

    def read(self) -> bytes:
        """Returns objects stored data.

        Returns:
            encoded data in bytes
        """
        if self.data is None:
            raise ValueError("No data.")
        return self.data

    def save(self, data: bytes) -> None:
        """Saves argument data in class attribute.

        Arguments:
            data: data to be saved in bytes
        """
        if not isinstance(data, bytes):
            raise TypeError("Data should be in bytes.")
        self.data = data

    @property
    def id(self) -> str:
        """Asset id getter method.

        Returns:
            id (str): id={base64(asset_path)}-{version}
        """
        base64str = base64.b64encode(self.asset_path.encode()).decode()
        return f'{base64str}-{self.version}'
