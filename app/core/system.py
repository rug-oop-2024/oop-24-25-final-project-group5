from typing import List

from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import LocalStorage
from autoop.core.storage import Storage


class ArtifactRegistry():
    """
    This is an artifact registry which uses a storage implementation
    and a database implementation.
    """
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        Initializes the registry with a database and storage backend.

        Args:
            database (Database): The database for storing metadata.
            storage (Storage): The storage for saving artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers an artifact by saving its data and metadata.

        Args:
            artifact (Artifact): The artifact to be registered.
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists artifacts of a specific type, if provided.

        Args:
            type (str, optional): The type of artifacts to list.

        Returns:
            List[Artifact]: List of artifacts of the specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def list_of_type[T](self, type_class: type[T], list_type: str = None) -> List[T]:  # noqa
        """
        Lists artifacts of a specified class and optional type.

        Args:
            type_class (type[T]): The class of artifacts to list.
            list_type (str, optional): Specific artifact type to filter.

        Returns:
            List[T]: List of artifacts of the specified class and type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if list_type is not None and data["type"] != list_type:
                continue
            artifact = type_class(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact by its ID.

        Args:
            artifact_id (str): The ID of the artifact to retrieve.

        Returns:
            Artifact: The artifact with the specified ID.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact by its ID from storage and database.

        Args:
            artifact_id (str): The ID of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Main System of the program"""

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initializes the AutoML system with a storage and database.

        Args:
            storage (LocalStorage): Storage for artifact data.
            database (Database): Database for artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> 'AutoMLSystem':
        """
        Returns the singleton instance of AutoMLSystem, initializing it
        if necessary.

        Returns:
            AutoMLSystem: The singleton instance of the AutoML system.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Provides access to the artifact registry.
metri
        Returns:
            ArtifactRegistry: The artifact registry instance.
        """
        return self._registry
