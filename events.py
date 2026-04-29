from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time

@dataclass
class Event:
    """Base class for all system events in the Sociogenesis simulation."""
    event_type: str
    source_agent_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class TransactionEvent(Event):
    """Event triggered during an economic transaction."""
    event_type: str = "TRANSACTION"
    amount: float = 0.0

@dataclass
class CommunicationEvent(Event):
    """Event triggered when agents communicate."""
    event_type: str = "COMMUNICATION"
    message: str = ""
