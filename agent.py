import asyncio
import random

class BaseAgent:
    def __init__(self, agent_id, economy, registry, memory, bus, tasks):
        self.id = agent_id
        self.economy = economy
        self.registry = registry
        self.memory = memory
        self.bus = bus
        self.tasks = tasks
        self.running = False

    async def run(self, max_ticks: int):
        self.running = True
        self.bus.register(self.id)
        
        for tick in range(max_ticks):
            if not self.running:
                break
                
            status = self.economy.spend(self.id, 1)
            if status is not None:
                print(f"[Tick {tick}] Agent {self.id} hit 0 tokens. Bailout received!")
                
            try:
                msg = await asyncio.wait_for(self.bus.receive(self.id), timeout=0.001)
            except asyncio.TimeoutError:
                pass
                
            rand = random.random()
            if rand < 0.2:
                self.economy.earn(self.id, 3)
            elif rand < 0.3:
                self.economy.earn(self.id, 5)
                
            await asyncio.sleep(0)
            
        self.running = False
