import asyncio
from typing import Dict, Any, Optional

class ArtifactStore:
    """
    ArtifactStore manages the persistence and retrieval of agents' creations,
    logs, and other generated materials within the Sociogenesis environment.
    """
    def __init__(self):
        # In-memory dictionary representing the storage, can be backed by SQLite later.
        self._store: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def save_artifact(self, artifact_id: str, content: Any) -> bool:
        """Saves an artifact asynchronously."""
        async with self._lock:
            self._store[artifact_id] = content
            return True

    async def get_artifact(self, artifact_id: str) -> Optional[Any]:
        """Retrieves an artifact asynchronously."""
        async with self._lock:
            return self._store.get(artifact_id)

    async def delete_artifact(self, artifact_id: str) -> bool:
        """Deletes an artifact asynchronously."""
        async with self._lock:
            if artifact_id in self._store:
                del self._store[artifact_id]
                return True
            return False

    async def list_artifacts(self) -> list:
        """Returns a list of all current artifact IDs."""
        async with self._lock:
            return list(self._store.keys())
