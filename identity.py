import uuid
import hashlib
from typing import Dict, Optional
import time

class IdentityManager:
    """
    Manages secure identities for agents in the Sociogenesis simulation.
    Generates unique IDs and handles identity verification.
    """
    def __init__(self):
        self.identities: Dict[str, dict] = {}

    def generate_identity(self, prefix: str = "agent") -> str:
        """Generates a new unique identity and records its creation time."""
        unique_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()
        
        # Create a simple hash-based token for internal verification
        token_data = f"{unique_id}:{timestamp}".encode('utf-8')
        verification_token = hashlib.sha256(token_data).hexdigest()
        
        self.identities[unique_id] = {
            "created_at": timestamp,
            "status": "active",
            "token": verification_token
        }
        
        return unique_id

    def verify_identity(self, agent_id: str, token: str) -> bool:
        """Verifies if an agent's identity token is valid."""
        if agent_id not in self.identities:
            return False
            
        record = self.identities[agent_id]
        if record["status"] != "active":
            return False
            
        return record["token"] == token

    def revoke_identity(self, agent_id: str) -> bool:
        """Revokes an agent's identity."""
        if agent_id in self.identities:
            self.identities[agent_id]["status"] = "revoked"
            return True
        return False
        
    def get_identity_info(self, agent_id: str) -> Optional[dict]:
        """Retrieves public information about an identity."""
        if agent_id in self.identities:
            record = self.identities[agent_id].copy()
            # Do not expose the private verification token
            record.pop("token", None)
            return record
        return None
