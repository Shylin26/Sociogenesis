"""
Custom exceptions for the Sociogenesis multi-agent system.
"""

class SociogenesisError(Exception):
    """Base exception for all Sociogenesis related errors."""
    pass

class EconomyError(SociogenesisError):
    """Raised when an economy-related operation fails."""
    pass

class InsufficientTokensError(EconomyError):
    """Raised when an agent tries to spend more tokens than they have."""
    pass

class CommunicationError(SociogenesisError):
    """Raised when a message cannot be sent or received."""
    pass

class RegistryError(SociogenesisError):
    """Raised when there is an issue with agent registration or lookup."""
    pass

class AgentNotRegisteredError(RegistryError):
    """Raised when an operation is performed on an unregistered agent."""
    pass
