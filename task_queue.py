import asyncio
from typing import Dict, Any

class TaskQueue:
    def __init__(self):
        self.queue = asyncio.PriorityQueue()
        self.task_counter = 0

    async def add_task(self, priority: int, task: Dict[str, Any]):
        self.task_counter += 1
        await self.queue.put((priority, self.task_counter, task))

    async def get_task(self):
        return await self.queue.get()

    def qsize(self):
        return self.queue.qsize()

    def is_empty(self):
        return self.queue.empty()
