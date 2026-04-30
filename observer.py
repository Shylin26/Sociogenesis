from typing import List
from events import Event

class SimulationObserver:
    """Observer pattern interface for monitoring the simulation."""
    
    def on_event(self, event: Event) -> None:
        """Called when an event occurs in the simulation."""
        pass

class EventLoggerObserver(SimulationObserver):
    """Logs all simulation events to memory."""
    
    def __init__(self):
        self.events: List[Event] = []
        
    def on_event(self, event: Event) -> None:
        """Store the event."""
        self.events.append(event)
