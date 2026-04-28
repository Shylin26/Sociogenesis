"""
Utility functions and helpers for the Sociogenesis multi-agent system.
"""
import uuid
import time
from typing import Any, Dict

def generate_id() -> str:
    """
    Generate a unique identifier, commonly used for agents, tasks, and messages.
    """
    return str(uuid.uuid4())

def current_timestamp_ms() -> int:
    """
    Return the current system time in milliseconds.
    """
    return int(time.time() * 1000)

def format_agent_stats(agent_id: str, balance: int, status: str) -> str:
    """
    Format basic agent statistics into a readable string for logging or UI display.
    """
    return f"Agent {agent_id[:8]}... | Status: {status} | Balance: {balance} tokens"

def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from a dictionary.
    """
    if d is None:
        return default
    return d.get(key, default)
