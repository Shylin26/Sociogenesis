import asyncio

class CommunicationBus:
    def __init__(self):
        self.queues = {}
        
    def register(self, agent_id: int):
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.Queue()
            
    async def send(self, sender_id: int, receiver_id: int, message: any):
        if receiver_id in self.queues:
            await self.queues[receiver_id].put({
                "sender": sender_id,
                "msg": message,
                "type": "direct"
            })
            
    async def broadcast(self, sender_id: int, message: any):
        msg_obj = {
            "sender": sender_id,
            "msg": message,
            "type": "broadcast"
        }
        for q in self.queues.values():
            await q.put(msg_obj)
            
    async def receive(self, agent_id: int):
        if agent_id in self.queues:
            return await self.queues[agent_id].get()
        return None
