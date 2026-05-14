import asyncio
import time
from dataclasses import dataclass,field
from typing import Any , Callable ,Coroutine,Optional
from collections import defaultdict

@dataclass
class Signal:
    tag : str
    data : Any
    tick : int
    sender_id : Any=None
    signal_id: int =field(default_factory=lambda: Signal._next_id())
    timestamp : float=field(default_factory=time.monotonic)

    _counter : int =0
    @staticmethod
    def _next_id()->int:
        Signal._counter+=1
        return Signal._counter
    
    def to_dict(self)->dict:
        return{
            "signal_id": self.signal_id,
            "tag":self.tag,
            "data": self.data,
            "tick": self.tick,
            "sender_id": self.sender_id,
            "timestamp":self.timestamp,
        }

@dataclass
class _Subscriber:
    callback: Callable[[Signal],Coroutine]
    name: str
    once : bool =False

class CommunicationBus:
    def __init__ (self,maxsize:int =0,history_cap: int =10_000):
        self._subscribers:dict[str,list[_Subscriber]]=defaultdict(list)
        self._queue: asyncio.Queue=asyncio.Queue(maxsize=maxsize)
        self._wildcard: list[_Subscriber]=[]
        self.history:list[Signal]=[]
        self.history_cap: int=history_cap
        self.total_published: int=0
        self.total_delivered: int =0
        self.total_dropped: int =0
        self._running : bool = False
    
    def subscribe(self,tag: str, callback : Callable,name: str="unnamed",once:bool = False):
        sub=_Subscriber(callback=callback,name=name,once=once)
        if tag =="*":
            self._wildcard.append(sub)
        else:
            self._subscribers[tag].append(sub)
    
    def unsubscribe(self,tag: str,name: str):
        """Remove all subscriptions matching name under tag."""
        if tag =="*":
            self._wildcard=[s for s in self._wildcard if s.name != name]
        else:
            self._subscribers[tag]=[
                s for s in self._subscribers[tag] if s.name!=name 
            ]
        
    def subscription_count(self,tag:str)->int:
        return len(self._subscribers.get(tag,[]))
    
    
    async def publish(self,signal:Signal):
        await self._queue.put(signal)
        self.total_published+=1
        self._record(signal)
    
    def publish_sync(self,signal:Signal):
        self._queue.put_nowait(signal)
        self.total_published+=1
        self._record(signal)
    
    async def run(self):
        self._running=True
        while( self._running):
            try:
                signal=await asyncio.wait_for(
                    self._queue.get(),timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            await self._deliver(signal)
            self._queue.task_done()
    
    def stop(self):
        self._running=False
    
    async def _deliver(self,signal:Signal):
        tagged=self._subscribers.get(signal.tag,[])
        all_subs=list(tagged)+list(self._wildcard)

        if not all_subs:
            self.total_dropped+=1
            return
        tasks=[sub.callback(signal) for sub in all_subs]
        results=await asyncio.gather(
            *tasks,return_exceptions=True

        )
        for sub, result in zip(all_subs,results):
            if isinstance(result,Exception):
                print(f"[Bus] '{sub.name}' raised on '{signal.tag}': {result}")
        
        self.total_delivered=len(all_subs)

        for sub in all_subs:
            if sub.once:
                self.unsubscribe(signal.tag,sub.name)
    
    def _record(self,signal:Signal):
        if len(self.history)>=self.history_cap:
            self.history.pop(0)
        self.history.append(signal)
    
    def snapshot(self)->dict:
        return{
            "queue_depth":self._queue.qsize(),
            "total_published":self.total_published,
            "total_delivered":self.total_published,
            "total_dropped":self.total_dropped,
            "subscribers"     : {
                tag: [s.name for s in subs]
                for tag, subs in self._subscribers.items()
            },

        }
    
    def recent(self,n:int=10,tag:Optional[str]=None)->list[dict]:
        signals=self.history
        if tag:
            signals=[s for s in signals if s.tag==tag]
        return [s.to_dict() for s in signals[-n:]]

class Tag:
    TASK_POSTED="task_posted"
    TASK_CLAIMED="task_claimed"
    TASK_COMPLETED="task_completed"
    TASK_FAILED="task_failed"
    TASK_EXPIRED="task_expired"

    AGENT_BORN="agent_born"
    AGENT_DIED="agent_died"
    
    TOKENS_EARNED="tokens_earned"
    TOKENS_SPENT="tokens_spent"

    FINGERPRINT_UPDATED="fingerprint_updated"
    CLUSTER_FORMED="cluster_formed"

    COALITION_FORMED="coalition_formed"
    COALITION_DISSOLVED="coalition_dissolved"

    ARTIFACT_SAVED="artifact_saved"

    EVOLUTION_TICK="evolution_tick"
    BENCHMARK_RUN="benchmark_run"


async def _smoke_test():
    print("=== CommunicationBus smoke test ===\n")
    bus=CommunicationBus()
    loop_task=asyncio.create_task(bus.run())

    received=[]
    async def on_completed(signal:Signal):
        received.append(signal.data["reward"])
    bus.subscribe(Tag.TASK_COMPLETED, on_completed, name="economy")
    await bus.publish(Signal(Tag.TASK_COMPLETED, {"agent_id":3,"reward":18}, tick=1))
    await asyncio.sleep(0.05)
    assert received==[18]
    print(" base publish/subscribe")

    reg_hits=[]
    async def on_completed_reg(signal:Signal):
        reg_hits.append(signal.data["agent_id"])
    bus.subscribe(Tag.TASK_COMPLETED,on_completed_reg,name="registry")

    await bus.publish(Signal(Tag.TASK_COMPLETED,{"agent_id":7,"reward":8},tick=2))
    await asyncio.sleep(0.05)
    assert 8 in received
    assert 7 in reg_hits
    print("multiple subscribers on the same tag")

    wild_tags=[]

    async def wildcard(signal:Signal):
        wild_tags.append(signal.tag)
    
    bus.subscribe("*",wildcard,name="dashboard")

    await bus.publish(Signal(Tag.AGENT_DIED,{"agent_id":2},tick =3))
    await bus.publish(Signal(Tag.TOKENS_EARNED,{"agent_id":5},tick =3))
    await asyncio.sleep(0.05)

    assert Tag.AGENT_DIED in wild_tags
    assert Tag.TOKENS_EARNED in wild_tags
    print("wildcard. recieved all tags")

    count=[0]
    async def once_cb(signal:Signal):
        count[0]+=1

    bus.subscribe(Tag.AGENT_BORN,once_cb,name="one_shot",once=True)
    await bus.publish(Signal(Tag.AGENT_BORN, {"agent_id": 9}, tick=4))
    await bus.publish(Signal(Tag.AGENT_BORN, {"agent_id": 9}, tick=5))
    await asyncio.sleep(0.05)
    assert count[0] == 1
    print("once=True fires only once")

    bus.unsubscribe("*","dashboard")
    pre_dropped=bus.total_dropped
    await bus.publish(Signal("orphan_tag", {}, tick=6))
    await asyncio.sleep(0.05)
    assert bus.total_dropped == pre_dropped + 1
    print("signals with no subscribers are dropped")

    print(f"\nSnapshot : {bus.snapshot()}")
    print(f"History  : {len(bus.history)} entries")
    print(f"Recent   : {[s['tag'] for s in bus.recent(5)]}")
    
    bus.stop()
    await loop_task
    print("\n=== All assertions passed ===")

if __name__ =="__main__":
    asyncio.run(_smoke_test())
        














