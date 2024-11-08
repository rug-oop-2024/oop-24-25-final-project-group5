import base64


class Artifact():
    """Artifact object used to store assets
    containing specific asset information.
    """

    def __init__(self,
                 name: str,
                 data: bytes,
                 version: str = "1.0.0",
                 asset_path: str = "",
                 tags: list = [],
                 metadata: dict = {},
                 type: str = "") -> None:
        self.name = name
        self.version = version
        self.asset_path = asset_path
        self.tags = tags
        self.metadata = metadata
        self.data = data
        self.type = type

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
