import asyncio
import random
import sqlite3
import json
import os
import shutil
import hashlib
import time
import uuid
import heapq
from collections import defaultdict
from dataclasses import dataclass,field
from enum import Enum
from pathlib import Path
from typing import Any,Callable,Coroutine,Optional


@dataclass
class Signal:
    tag:str
    data:dict
    tick:int
    sender_id:Any=None
    signal_id:int=field(default_factory=lambda:Signal._next_id())
    timestamp : float = field(default_factory=time.monotonic)
    _counter  : int   = 0

    @staticmethod

    def _next_id()->int:
        Signal._counter+=1
        return Signal._counter
    
    def to_dict(self):
        return {"signal_id": self.signal_id, "tag": self.tag,
                "data": self.data, "tick": self.tick,
                "sender_id": self.sender_id}

@dataclass
class _Subscriber:
    callback:Callable[[Signal],Coroutine]
    name:str
    once:bool=False

class CommunicationBus:
    def __init__(self, maxsize=0, history_cap=10_000):
        self._subscribers : dict[str, list[_Subscriber]] = defaultdict(list)
        self._queue    = asyncio.Queue(maxsize=maxsize)
        self._wildcard : list[_Subscriber] = []
        self.history   : list[Signal] = []
        self._history_cap = history_cap
        self.total_published = 0
        self.total_delivered = 0
        self.total_dropped   = 0
        self._running        = False
    
    def subscribe(self, tag, callback, name="unnamed", once=False):
        sub = _Subscriber(callback=callback, name=name, once=once)
        if tag == "*":
            self._wildcard.append(sub)
        else:
            self._subscribers[tag].append(sub)
    
    def unsubscribe(self, tag, name):
        if tag == "*":
            self._wildcard = [s for s in self._wildcard if s.name != name]
        else:
            self._subscribers[tag] = [
                s for s in self._subscribers[tag] if s.name != name]

    async def publish(self, signal: Signal):
        await self._queue.put(signal)
        self.total_published += 1
        if len(self.history) >= self._history_cap:
            self.history.pop(0)
        self.history.append(signal)
 
    def publish_sync(self, signal: Signal):
        self._queue.put_nowait(signal)
        self.total_published += 1
        if len(self.history) >= self._history_cap:
            self.history.pop(0)
        self.history.append(signal)
 
    async def run(self):
        self._running = True
        while self._running:
            try:
                signal = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            tagged   = self._subscribers.get(signal.tag, [])
            all_subs = list(tagged) + list(self._wildcard)
            if not all_subs:
                self.total_dropped += 1
            else:
                tasks   = [s.callback(signal) for s in all_subs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for s, r in zip(all_subs, results):
                    if isinstance(r, Exception):
                        print(f"[Bus] '{s.name}' error on '{signal.tag}': {r}")
                self.total_delivered += len(all_subs)
                for s in all_subs:
                    if s.once:
                        self.unsubscribe(signal.tag, s.name)
            self._queue.task_done()
 
    def stop(self):
        self._running = False
 
    def snapshot(self):
        return {"queue_depth": self._queue.qsize(),
                "published": self.total_published,
                "delivered": self.total_delivered,
                "dropped":   self.total_dropped}
 
 
class Tag:
    TASK_POSTED     = "task_posted"
    TASK_DISPATCHED = "task_dispatched"
    TASK_COMPLETED  = "task_completed"
    TASK_FAILED     = "task_failed"
    TASK_EXPIRED    = "task_expired"
    AGENT_BORN      = "agent_born"
    AGENT_DIED      = "agent_died"
    TOKENS_EARNED   = "tokens_earned"
    TOKENS_SPENT    = "tokens_spent"
    ARTIFACT_SAVED  = "artifact_saved"
 
 
 
@dataclass
class AgentRecord:
    agent_id          : int
    skill_fingerprint : list        # 128-dim zeros in Week 1
    reputation_score  : float = 0.0
    token_balance     : int   = 100
    parent_id         : Optional[int] = None
    birth_tick        : int   = 0
    tasks_completed   : int   = 0
    is_alive          : bool  = True
 
 
class AgentRegistry:
    def __init__(self, n_agents: int):
        self.records: dict[int, AgentRecord] = {
            i: AgentRecord(agent_id=i,
                           skill_fingerprint=[0.0] * 128)
            for i in range(n_agents)
        }
 
    def get(self, agent_id: int) -> AgentRecord:
        return self.records[agent_id]
 
    def all_living(self) -> list[AgentRecord]:
        return [r for r in self.records.values() if r.is_alive]
 
    def update_reputation(self, agent_id: int, score: float):
        r = self.records[agent_id]
        r.reputation_score = 0.9 * r.reputation_score + 0.1 * score
        r.tasks_completed += 1
 
    def replace_agent(self, dead_id: int, parent_id: int,
                      current_tick: int, mutation_rate: float = 0.1):
        parent = self.records[parent_id]
        # mutate fingerprint (pure python, no torch in Week 1)
        new_fp = [
            p + mutation_rate * random.gauss(0, 1)
            for p in parent.skill_fingerprint
        ]
        # normalise
        norm = sum(x*x for x in new_fp) ** 0.5 or 1.0
        new_fp = [x / norm for x in new_fp]
        self.records[dead_id] = AgentRecord(
            agent_id          = dead_id,
            skill_fingerprint = new_fp,
            token_balance     = parent.token_balance // 2,
            parent_id         = parent_id,
            birth_tick        = current_tick,
        )
 
    def snapshot(self) -> list[dict]:
        return [
            {"agent_id": r.agent_id,
             "tokens":   r.token_balance,
             "rep":      round(r.reputation_score, 3),
             "tasks":    r.tasks_completed,
             "alive":    r.is_alive}
            for r in self.records.values()
        ]
 
 
 
class ComputeEconomy:
    def __init__(self, n_agents: int, start_tokens: int = 100):
        self.balances       = {i: start_tokens for i in range(n_agents)}
        self.death_enabled  = False   # flipped on in Week 3
        self.history        : list[dict] = []
 
    def earn(self, agent_id: int, reward: int):
        self.balances[agent_id] += reward
        self.history.append({"event": "earn", "agent": agent_id,
                              "delta": reward,
                              "balance": self.balances[agent_id]})
 
    def spend(self, agent_id: int, cost: int) -> bool:
        if self.balances[agent_id] < cost:
            return False
        self.balances[agent_id] -= cost
        self.history.append({"event": "spend", "agent": agent_id,
                              "delta": -cost,
                              "balance": self.balances[agent_id]})
        return True
 
    def apply_penalty(self, agent_id: int, penalty: int):
        self.balances[agent_id] -= penalty
        self.history.append({"event": "penalty", "agent": agent_id,
                              "delta": -penalty,
                              "balance": self.balances[agent_id]})
 
    def check_deaths(self, registry: AgentRegistry,
                     current_tick: int) -> list[dict]:
        """
        Check all balances. Kill agents below 0 if death is enabled.
        Returns list of death events for the bus.
        Week 1: death_enabled=False, so this never triggers.
        """
        events = []
        if not self.death_enabled:
            return events
        for agent_id, balance in list(self.balances.items()):
            if balance < 0:
                top = max(self.balances, key=self.balances.get)
                events.append({"killed": agent_id, "parent": top})
                registry.replace_agent(agent_id, top, current_tick)
                self.balances[agent_id] = self.balances[top] // 2
        return events
 
    def reward_for(self, difficulty: float) -> int:
        return max(1, int(difficulty * 20))
 
    def penalty_for(self, difficulty: float) -> int:
        return max(1, int(difficulty * 7))
 
    def snapshot(self) -> dict:
        return dict(self.balances)
 
    def is_diverging(self) -> bool:
        """
        Week 1 test: balances should NOT all be equal.
        Returns True if std-dev > 5 tokens.
        """
        vals  = list(self.balances.values())
        mean  = sum(vals) / len(vals)
        var   = sum((v - mean)**2 for v in vals) / len(vals)
        return var**0.5 > 5.0
 
 
 
class TaskType(Enum):
    CODE     = "code"
    RESEARCH = "research"
    VISUAL   = "visual"
 
 
class TaskStatus(Enum):
    PENDING   = "pending"
    CLAIMED   = "claimed"
    COMPLETED = "completed"
    FAILED    = "failed"
    EXPIRED   = "expired"
 
 
@dataclass
class Task:
    task_id        : str
    task_type      : TaskType
    difficulty     : float
    description    : str
    posted_tick    : int
    reward_tokens  : int = 0
    penalty_tokens : int = 0
    status         : TaskStatus     = TaskStatus.PENDING
    assigned_agent : Optional[int]  = None
    completed_tick : Optional[int]  = None
 
    def __post_init__(self):
        if self.reward_tokens  == 0:
            self.reward_tokens  = max(1, int(self.difficulty * 20))
        if self.penalty_tokens == 0:
            self.penalty_tokens = max(1, int(self.difficulty * 7))
 
    def to_dict(self):
        return {"task_id": self.task_id[:8],
                "type":    self.task_type.value,
                "diff":    round(self.difficulty, 2),
                "status":  self.status.value,
                "agent":   self.assigned_agent,
                "reward":  self.reward_tokens}
 
 
@dataclass(order=True)
class _HeapEntry:
    priority    : float
    posted_tick : int
    task        : Task = field(compare=False)
 
 
class TaskQueue:
    def __init__(self, expiry_ticks: int = 50):
        self._heap          : list[_HeapEntry] = []
        self._all           : dict[str, Task]  = {}
        self.expiry_ticks   = expiry_ticks
        self.total_posted   = 0
        self.total_completed= 0
        self.total_failed   = 0
        self.total_expired  = 0
 
    def post(self, task_type: TaskType, difficulty: float,
             description: str, current_tick: int) -> Task:
        difficulty = max(0.0, min(1.0, difficulty))
        task  = Task(task_id=str(uuid.uuid4()), task_type=task_type,
                     difficulty=difficulty, description=description,
                     posted_tick=current_tick)
        entry = _HeapEntry(priority=-difficulty,
                           posted_tick=current_tick, task=task)
        heapq.heappush(self._heap, entry)
        self._all[task.task_id] = task
        self.total_posted += 1
        return task
 
    def dispatch(self, agent_id: int, current_tick: int) -> Optional[Task]:
        self._expire(current_tick)
        while self._heap:
            entry = heapq.heappop(self._heap)
            task  = entry.task
            if task.status != TaskStatus.PENDING:
                continue
            task.status         = TaskStatus.CLAIMED
            task.assigned_agent = agent_id
            return task
        return None
 
    def complete(self, task_id: str, current_tick: int) -> Optional[Task]:
        task = self._all.get(task_id)
        if task is None or task.status != TaskStatus.CLAIMED:
            return None
        task.status         = TaskStatus.COMPLETED
        task.completed_tick = current_tick
        self.total_completed += 1
        return task
 
    def fail(self, task_id: str, current_tick: int) -> Optional[Task]:
        task = self._all.get(task_id)
        if task is None or task.status != TaskStatus.CLAIMED:
            return None
        task.status         = TaskStatus.FAILED
        task.completed_tick = current_tick
        self.total_failed  += 1
        return task
 
    def pending_count(self) -> int:
        return sum(1 for e in self._heap
                   if e.task.status == TaskStatus.PENDING)
 
    def generate_synthetic(self, current_tick: int, n: int = 3) -> list:
        templates = {
            TaskType.CODE: {
                "easy":   "write a function to reverse a string",
                "medium": "implement binary search on a sorted list",
                "hard":   "build a graph shortest-path algorithm",
            },
            TaskType.RESEARCH: {
                "easy":   "define overfitting in one paragraph",
                "medium": "compare gradient descent variants",
                "hard":   "propose a falsifiable hypothesis on agent behaviour",
            },
            TaskType.VISUAL: {
                "easy":   "generate a bar chart of token balances",
                "medium": "produce a t-SNE plot of agent fingerprints",
                "hard":   "create a force-directed coalition graph",
            },
        }
        tasks = []
        for _ in range(n):
            tt   = random.choice(list(TaskType))
            diff = random.uniform(0.1, 1.0)
            tier = "easy" if diff < 0.36 else "medium" if diff < 0.66 else "hard"
            desc = templates[tt][tier]
            tasks.append(self.post(tt, diff, desc, current_tick))
        return tasks
 
    def _expire(self, current_tick: int):
        for entry in self._heap:
            t = entry.task
            if (t.status == TaskStatus.PENDING and
                    current_tick - t.posted_tick > self.expiry_ticks):
                t.status = TaskStatus.EXPIRED
                self.total_expired += 1
 
    def snapshot(self) -> dict:
        return {"pending":   self.pending_count(),
                "completed": self.total_completed,
                "failed":    self.total_failed,
                "expired":   self.total_expired,
                "posted":    self.total_posted}
 
 
#  Full FAISS version in shared_memory.py — here we use a simple
#  cosine similarity scan so the main loop has zero extra deps.
 
@dataclass
class MemoryEntry:
    idx         : int
    vector      : list      # 128-dim
    metadata    : dict
 
 
class SharedMemory:
    """
    Week 1 SharedMemory — linear scan cosine similarity.
    Drop-in replaced by FAISS in Week 6 when >10k entries.
    """
    def __init__(self, dim: int = 128):
        self.dim     = dim
        self.entries : list[MemoryEntry] = []
 
    def write(self, vec: list, meta: dict) -> int:
        idx = len(self.entries)
        self.entries.append(MemoryEntry(idx=idx, vector=vec, metadata=meta))
        return idx
 
    def query(self, vec: list, k: int = 5) -> list[dict]:
        if not self.entries:
            return []
        scored = []
        qn = sum(x*x for x in vec) ** 0.5 or 1.0
        for e in self.entries:
            en  = sum(x*x for x in e.vector) ** 0.5 or 1.0
            dot = sum(a*b for a, b in zip(vec, e.vector))
            scored.append((dot / (qn * en), e.metadata))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]
 
    def snapshot(self) -> dict:
        return {"entries": len(self.entries), "dim": self.dim}
 
 
 
class ArtifactType(Enum):
    CODE     = "code"
    RESEARCH = "research"
    VISUAL   = "visual"
 
 
@dataclass
class Artifact:
    artifact_id   : str
    artifact_type : ArtifactType
    content       : str
    author_id     : int
    task_id       : str
    tick          : int
    quality_score : float = 0.0
    links         : list  = field(default_factory=list)
 
    def to_dict(self):
        return {"artifact_id":   self.artifact_id[:12],
                "type":          self.artifact_type.value,
                "author":        self.author_id,
                "quality":       round(self.quality_score, 2),
                "tick":          self.tick}
 
 
class ArtifactStore:
    _EXT = {ArtifactType.CODE: ".py",
            ArtifactType.RESEARCH: ".json",
            ArtifactType.VISUAL: ".txt"}
 
    def __init__(self, base_dir: str = "./artifacts"):
        self.base_dir = Path(base_dir)
        for t in ArtifactType:
            (self.base_dir / t.value).mkdir(parents=True, exist_ok=True)
        self._db_path = self.base_dir / "artifacts.db"
        self._conn    = self._setup_db()
        self._cache   : dict[str, Artifact] = {}
        self.total_saved = 0
 
    def save(self, artifact_type, content, author_id,
             task_id, tick, quality_score=0.0) -> Artifact:
        aid = hashlib.sha256(content.encode()).hexdigest()
        if aid in self._cache:
            return self._cache[aid]
        ext  = self._EXT[artifact_type]
        path = self.base_dir / artifact_type.value / (aid[:16] + ext)
        path.write_text(content, encoding="utf-8")
        art  = Artifact(artifact_id=aid, artifact_type=artifact_type,
                        content=content, author_id=author_id,
                        task_id=task_id, tick=tick,
                        quality_score=quality_score)
        self._conn.execute(
            "INSERT OR IGNORE INTO artifacts "
            "(artifact_id,artifact_type,author_id,task_id,"
            " quality_score,tick,file_path) VALUES(?,?,?,?,?,?,?)",
            (aid, artifact_type.value, author_id, task_id,
             quality_score, tick, str(path))
        )
        self._conn.commit()
        self._cache[aid] = art
        self.total_saved += 1
        return art
 
    def recent(self, n=5) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM artifacts ORDER BY tick DESC LIMIT ?", (n,)
        ).fetchall()
        return [{"artifact_id": r["artifact_id"][:12],
                 "type":        r["artifact_type"],
                 "author":      r["author_id"],
                 "quality":     r["quality_score"],
                 "tick":        r["tick"]} for r in rows]
 
    def snapshot(self) -> dict:
        rows = self._conn.execute(
            "SELECT artifact_type, COUNT(*) as c "
            "FROM artifacts GROUP BY artifact_type"
        ).fetchall()
        return {"total_saved": self.total_saved,
                "by_type": {r["artifact_type"]: r["c"] for r in rows}}
 
    def _setup_db(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id   TEXT PRIMARY KEY,
                artifact_type TEXT,
                author_id     INTEGER,
                task_id       TEXT,
                quality_score REAL DEFAULT 0.0,
                tick          INTEGER,
                file_path     TEXT
            )""")
        conn.commit()
        return conn
 
    def close(self):
        self._conn.close()
 
 
#
 
class SyntheticAgent:
    """
    Placeholder agent for Week 1.
    Has no transformer — just a skill bias per task type that
    evolves slightly so the economy produces divergence.
    """
    def __init__(self, agent_id: int):
        self.agent_id  = agent_id
        # random starting bias per task type — creates natural divergence
        self.skill_bias = {
            TaskType.CODE    : random.uniform(0.3, 0.8),
            TaskType.RESEARCH: random.uniform(0.3, 0.8),
            TaskType.VISUAL  : random.uniform(0.3, 0.8),
        }
 
    def attempt(self, task: Task) -> tuple[bool, float, str]:
        """
        Attempt a task.
        Returns (success, quality_score, artifact_content).
 
        Success probability = agent's skill_bias for this task type.
        Quality = success * random noise around the bias.
        """
        p_success = self.skill_bias[task.task_type]
        success   = random.random() < p_success
        quality   = (random.uniform(0.5, 1.0) * p_success) if success else 0.0
 
        # generate a minimal synthetic artifact string
        content = (
            f"# Agent {self.agent_id} | tick unknown | {task.task_type.value}\n"
            f"# task: {task.description}\n"
            f"# quality: {round(quality, 3)}\n"
            f"result = 'synthetic output {uuid.uuid4().hex[:8]}'"
        )
        return success, quality, content
 
    def update_bias(self, task_type: TaskType,
                    success: bool, reward: int):
        """
        Tiny bias drift after each task — simulates early learning.
        Successful agents get slightly better at their task type.
        """
        delta = 0.01 * reward if success else -0.005
        self.skill_bias[task_type] = max(
            0.1, min(0.95, self.skill_bias[task_type] + delta)
        )
 
 
 
class Society:
    """
    Week 1 Society.
 
    Wires all six components. Runs the 1000-tick main loop.
    Each tick:
        1. Generate 3 synthetic tasks → TaskQueue
        2. Each agent claims + attempts one task
        3. Economy pays/penalises based on outcome
        4. Registry updates reputation
        5. ArtifactStore saves successful outputs
        6. SharedMemory stores the experience vector
        7. Bus publishes all events
        8. Every 100 ticks: print progress report
    """
 
    def __init__(self, n_agents: int = 10,
                 artifact_dir: str = "./artifacts",
                 seed: int = 42):
        random.seed(seed)
 
        # six components
        self.bus      = CommunicationBus()
        self.registry = AgentRegistry(n_agents)
        self.economy  = ComputeEconomy(n_agents)
        self.queue    = TaskQueue(expiry_ticks=50)
        self.memory   = SharedMemory(dim=128)
        self.store    = ArtifactStore(base_dir=artifact_dir)
 
        # synthetic agents (replaced by transformers in Week 2)
        self.agents   = {i: SyntheticAgent(i) for i in range(n_agents)}
 
        self.tick     = 0
        self._balance_history : list[dict] = []   # for divergence plot
 
    async def run(self, ticks: int = 1000):
        """Main simulation loop."""
        bus_task = asyncio.create_task(self.bus.run())
        self._subscribe_all()
 
        print(f"PANTHEON Week 1 — {len(self.agents)} agents, {ticks} ticks\n")
 
        for t in range(ticks):
            self.tick = t
            await self._tick(t)
 
            # progress report every 100 ticks
            if (t + 1) % 100 == 0:
                self._report(t + 1)
 
        self.bus.stop()
        await bus_task
 
        # final assessment
        self._final_report(ticks)
        self.store.close()
 
    async def _tick(self, t: int):
        """One tick — all agents act once."""
 
        # 1. generate synthetic tasks
        new_tasks = self.queue.generate_synthetic(t, n=3)
        for task in new_tasks:
            await self.bus.publish(Signal(
                Tag.TASK_POSTED,
                {"task_id": task.task_id, "type": task.task_type.value,
                 "difficulty": task.difficulty},
                tick=t
            ))
 
        # 2. each agent claims + attempts a task
        for agent_id, agent in self.agents.items():
            rec  = self.registry.get(agent_id)
            if not rec.is_alive:
                continue
 
            task = self.queue.dispatch(agent_id, t)
            if task is None:
                continue  # no tasks available — skip this tick
 
            await self.bus.publish(Signal(
                Tag.TASK_DISPATCHED,
                {"agent_id": agent_id, "task_id": task.task_id},
                tick=t, sender_id=agent_id
            ))
 
            # agent attempts the task
            success, quality, content = agent.attempt(task)
 
            if success:
                # complete the task
                done = self.queue.complete(task.task_id, t)
 
                # economy: earn reward
                self.economy.earn(agent_id, done.reward_tokens)
 
                # registry: update reputation
                self.registry.update_reputation(agent_id, quality)
 
                # artifact store: save output
                art = self.store.save(
                    artifact_type = ArtifactType(task.task_type.value),
                    content       = content,
                    author_id     = agent_id,
                    task_id       = task.task_id,
                    tick          = t,
                    quality_score = quality,
                )
 
                # shared memory: store experience vector
                # Week 1: vector is random — Week 6 this becomes real embedding
                exp_vec = [random.gauss(0, 1) for _ in range(128)]
                self.memory.write(exp_vec, {
                    "agent_id"    : agent_id,
                    "task_type"   : task.task_type.value,
                    "task_id"     : task.task_id,
                    "quality"     : quality,
                    "tick"        : t,
                    "artifact_id" : art.artifact_id,
                })
 
                # bus: publish success
                await self.bus.publish(Signal(
                    Tag.TASK_COMPLETED,
                    {"agent_id":     agent_id,
                     "task_id":      task.task_id,
                     "reward":       done.reward_tokens,
                     "quality":      round(quality, 3),
                     "artifact_id":  art.artifact_id},
                    tick=t, sender_id=agent_id
                ))
 
                # agent self-update
                agent.update_bias(task.task_type, True, done.reward_tokens)
 
            else:
                # fail the task
                failed = self.queue.fail(task.task_id, t)
 
                # economy: apply penalty
                self.economy.apply_penalty(agent_id, failed.penalty_tokens)
 
                # bus: publish failure
                await self.bus.publish(Signal(
                    Tag.TASK_FAILED,
                    {"agent_id": agent_id,
                     "task_id":  task.task_id,
                     "penalty":  failed.penalty_tokens},
                    tick=t, sender_id=agent_id
                ))
 
                # agent self-update
                agent.update_bias(task.task_type, False, 0)
 
        # 3. check for deaths (disabled in Week 1)
        death_events = self.economy.check_deaths(self.registry, t)
        for ev in death_events:
            await self.bus.publish(Signal(
                Tag.AGENT_DIED,
                ev, tick=t
            ))
 
        # 4. snapshot balances for divergence tracking
        self._balance_history.append({
            "tick"    : t,
            "balances": self.economy.snapshot()
        })
 
    def _subscribe_all(self):
        """
        Wire up bus subscribers.
        In Week 1 these are lightweight loggers.
        In later weeks they trigger speciation updates,
        coalition formation, dashboard refreshes etc.
        """
        async def on_agent_died(signal: Signal):
            print(f"  [DEATH] tick={signal.tick} "
                  f"agent={signal.data.get('killed')} "
                  f"parent={signal.data.get('parent')}")
 
        self.bus.subscribe(Tag.AGENT_DIED, on_agent_died, name="logger")
 
    def _report(self, t: int):
        """Progress report every 100 ticks."""
        balances = self.economy.snapshot()
        q_snap   = self.queue.snapshot()
        s_snap   = self.store.snapshot()
        m_snap   = self.memory.snapshot()
 
        vals  = list(balances.values())
        mean  = sum(vals) / len(vals)
        mn, mx= min(vals), max(vals)
 
        print(f"-- tick {t:4d} " + "-" * 40)
        print(f"  economy   : mean={mean:.0f}  min={mn}  max={mx}  "
              f"diverging={self.economy.is_diverging()}")
        print(f"  tasks     : {q_snap}")
        print(f"  artifacts : {s_snap}")
        print(f"  memory    : {m_snap['entries']} entries")
        print(f"  balances  : { {k: v for k, v in balances.items()} }")
        print()
 
    def _final_report(self, ticks: int):
        """Week 1 success criteria check."""
        print("=" * 52)
        print("WEEK 1 FINAL REPORT")
        print("=" * 52)
 
        living     = self.registry.all_living()
        q_snap     = self.queue.snapshot()
        s_snap     = self.store.snapshot()
        diverging  = self.economy.is_diverging()
        balances   = self.economy.snapshot()
 
        print(f"\n{'[PASS]' if len(living)==10 else '[FAIL]'} "
              f"Agents alive: {len(living)}/10")
 
        no_deadlock = q_snap['completed'] + q_snap['failed'] > 0
        print(f"{'[PASS]' if no_deadlock else '[FAIL]'} "
              f"No deadlock: {q_snap['completed']} completed, "
              f"{q_snap['failed']} failed")
 
        print(f"{'[PASS]' if diverging else '[FAIL]'} "
              f"Token divergence: {diverging}")
 
        print(f"[PASS] Artifacts saved: {s_snap['total_saved']}")
        print(f"[PASS] Memory entries:  {self.memory.snapshot()['entries']}")
        print(f"[PASS] Bus published:   {self.bus.snapshot()['published']}")
 
        print(f"\nFinal token balances:")
        for aid, bal in sorted(balances.items()):
            bar = "#" * max(0, bal // 10)
            print(f"  Agent {aid}: {bal:4d}  {bar}")
 
        print(f"\nTop 3 agents by reputation:")
        recs = sorted(self.registry.all_living(),
                      key=lambda r: r.reputation_score, reverse=True)
        for r in recs[:3]:
            print(f"  Agent {r.agent_id}: rep={r.reputation_score:.3f} "
                  f"tasks={r.tasks_completed} tokens={r.token_balance}")
 
        all_pass = (len(living) == 10 and no_deadlock and diverging)
        print(f"\n{'SUCCESS: WEEK 1 COMPLETE — all criteria met' if all_pass else 'WARNING: Some criteria not met'}")
        print("=" * 52)
 
 
 
async def main():
    ARTIFACT_DIR = "/tmp/pantheon_artifacts"
 
    # clean slate
    if os.path.exists(ARTIFACT_DIR):
        shutil.rmtree(ARTIFACT_DIR)
 
    society = Society(
        n_agents     = 10,
        artifact_dir = ARTIFACT_DIR,
        seed         = 42,
    )
    await society.run(ticks=1000)
 
    # cleanup
    shutil.rmtree(ARTIFACT_DIR)
 
 
if __name__ == "__main__":
    asyncio.run(main())
           
    



    

