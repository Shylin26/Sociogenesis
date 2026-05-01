"""
Security and permissions framework for the Sociogenesis multi-agent system.
Handles agent capabilities, access control, and capability verification.
"""

from dataclasses import dataclass
from typing import Set, List
import enum

class PermissionLevel(enum.Enum):
    GUEST = 0
    AGENT = 1
    ADMIN = 2
    SYSTEM = 3

@dataclass
class SecurityContext:
    agent_id: str
    level: PermissionLevel
    capabilities: Set[str]

class SecurityManager:
    def __init__(self):
        self._contexts = {}
        
    def register_agent(self, agent_id: str, level: PermissionLevel, capabilities: List[str] = None):
        if capabilities is None:
            capabilities = []
        self._contexts[agent_id] = SecurityContext(
            agent_id=agent_id,
            level=level,
            capabilities=set(capabilities)
        )
        
    def has_permission(self, agent_id: str, required_level: PermissionLevel) -> bool:
        context = self._contexts.get(agent_id)
        if not context:
            return False
        return context.level.value >= required_level.value
        
    def has_capability(self, agent_id: str, capability: str) -> bool:
        context = self._contexts.get(agent_id)
        if not context:
            return False
        return capability in context.capabilities or context.level == PermissionLevel.SYSTEM
