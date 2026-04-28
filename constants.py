"""
System-wide constants and enumerations for the Sociogenesis platform.
"""

from enum import Enum

class AgentState(Enum):
    IDLE = "IDLE"
    WORKING = "WORKING"
    COMMUNICATING = "COMMUNICATING"
    ERROR = "ERROR"

class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

DEFAULT_ENCODING = "utf-8"
MAX_RETRY_ATTEMPTS = 3
