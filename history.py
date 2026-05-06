from typing import List, Dict, Any
from datetime import datetime
from logger import logger

class HistoricalLedger:
    """
    Maintains a historical record of significant events and state changes 
    within the Sociogenesis simulation.
    """
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def record_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Records a new historical event with a timestamp.
        
        Args:
            event_type (str): The classification of the event.
            details (Dict[str, Any]): Additional context and data about the event.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "details": details
        }
        self._records.append(record)
        logger.debug(f"Recorded historical event: {event_type}")

    def get_history(self, event_type: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves the historical records, optionally filtered by event type.
        
        Args:
            event_type (str, optional): The type of events to filter by. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: A list of matching historical records.
        """
        if event_type:
            return [record for record in self._records if record["type"] == event_type]
        return self._records.copy()

    def clear_history(self) -> None:
        """Clears all historical records."""
        self._records.clear()
        logger.debug("Historical ledger cleared.")
