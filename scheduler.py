import asyncio
import time
from typing import Callable, Dict, Any, Awaitable

class Scheduler:
    """
    Manages timed events and periodic tasks for the simulation.
    Allows registering callbacks to be executed at specific intervals or times.
    """
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def schedule_periodic(self, name: str, interval: float, callback: Callable[..., Awaitable[Any]], *args, **kwargs):
        """Schedules an asynchronous callback to run periodically."""
        async def _periodic_worker():
            while self._running:
                await asyncio.sleep(interval)
                try:
                    await callback(*args, **kwargs)
                except Exception as e:
                    print(f"Error in scheduled task '{name}': {e}")
        
        self.tasks[name] = asyncio.create_task(_periodic_worker())

    def start(self):
        self._running = True

    def stop(self):
        self._running = False
        for task in self.tasks.values():
            task.cancel()
        self.tasks.clear()
