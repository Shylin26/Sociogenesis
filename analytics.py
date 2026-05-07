import typing
from datetime import datetime

class Analytics:
    """
    Analytics component for the Sociogenesis system.
    Responsible for tracking and reporting on simulation metrics and events over time.
    """
    def __init__(self):
        self.data_points = []
        self.metrics_snapshot = {}

    def record_event(self, event_type: str, data: typing.Dict):
        """
        Records a new event data point with the current timestamp.
        """
        self.data_points.append({
            "timestamp": datetime.now(),
            "type": event_type,
            "data": data
        })

    def generate_report(self) -> typing.Dict:
        """
        Generates a summary report of the recorded events.
        """
        return {
            "total_events": len(self.data_points),
            "events_by_type": self._aggregate_by_type(),
            "generated_at": datetime.now()
        }

    def _aggregate_by_type(self) -> typing.Dict[str, int]:
        """
        Aggregates the recorded events by their type.
        """
        aggregation = {}
        for dp in self.data_points:
            event_type = dp.get("type", "unknown")
            aggregation[event_type] = aggregation.get(event_type, 0) + 1
        return aggregation

    def clear_data(self):
        """
        Clears all recorded analytics data.
        """
        self.data_points = []
