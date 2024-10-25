from pydantic import BaseModel, Field, field_validator
import base64


class Artifact(BaseModel):
    name: str = Field()
    asset_path: str = Field()
    data: bytes = Field()
    version: str = Field()
    type: str = Field()

    def read(self) -> bytes:
        """
        Returns objects stored data.

        Returns:
            encoded data in bytes
        """
        if self.data is None:
            raise ValueError("No data.")
        return self.data

    def save(self, data: bytes) -> None:
        """
        Saves argument data in class attribute.

        Arguments:
            data: data to be saved in bytes
        """
        if not isinstance(data, bytes):
            raise TypeError("Data should be in bytes.")
        self.data = data
