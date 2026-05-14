import heapq
import uuid
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class TaskType(Enum):
    CODE="code"
    RESEARCH="research"
    VISUAL="visual"

class TaskStatus(Enum):
    PENDING="pending"
    CLAIMED="claimed"
    COMPLETED="completed"
    FAILED="failed"
    EXPIRED="expired"

@dataclass
class Task:
    task_id: str
    task_type: TaskType
    difficulty: float
    description: str
    posted_tick:int

    reward_tokens:int=0
    penalty_tokens:int=0

    status : TaskStatus=TaskStatus.PENDING
    assigned_agent: Optional[int]=None
    completed_tick: Optional[int]=None

    def __post_init__(self):
        if self.reward_tokens==0:
            self.reward_tokens=max(1,int(self.difficulty*20))
        if self.penalty_tokens==0:
            self.penalty_tokens=max(1,int(self.difficulty*7))
    
    def to_dict(self)->dict:
        return{
            "task_id":self.task_id,
            "task_type":self.task_type.value,
            "difficulty":self.difficulty,
            "description":self.description,
            "posted_tick":self.posted_tick,
            "reward_tokens":self.reward_tokens,
            "penalty_tokens":self.penalty_tokens,
            "status":self.status.value,
            "assigned_agent":self.assigned_agent,
            "completed_tick":self.completed_tick,
        }

@dataclass(order=True)
class _HeapEntry:
    priority : float
    posted_tick : int
    task: Task=field(compare=False)

class TaskQueue:
    def __init__(self,expiry_ticks:int=50):
        self._heap: list[_HeapEntry]=[]
        self._all :dict[str,Task]={}
        self.expiry_ticks=expiry_ticks
    
        self.total_posted=0
        self.total_failed=0
        self.total_completed=0
        self.total_expired=0

        self.history : list[dict]=[]
    
    def post(self,task_type: TaskType,difficulty: float,description:str,current_tick:int)->Task:
        difficulty=max(0.0,min(1.0,difficulty))
        task = Task(
            task_id     = str(uuid.uuid4()),
            task_type   = task_type,
            difficulty  = difficulty,
            description = description,
            posted_tick = current_tick,
        )
        entry=_HeapEntry(
            priority=-difficulty, # min heap serves highest first
            posted_tick=current_tick,
            task=task,
        )
        heapq.heappush(self._heap,entry)
        self._all[task.task_id]=task
        self.total_posted+=1

        self.log(current_tick,"posted",task.task_id,None,f"type={task.task_type.value} diff={difficulty:.2f} "f"reward={task.reward_tokens}")
        return task
    
    def dispatch(self,agent_id:int,current_tick:int)->Optional[Task]:
        self._expire(current_tick)

        while self._heap:
            entry=heapq.heappop(self._heap)
            task=entry.task
            if task.status !=TaskStatus.PENDING:
                continue
            task.status=TaskStatus.CLAIMED
            task.assigned_agent=agent_id
            self.log(current_tick,"dispacthed",task.task_id,agent_id,f"diff={task.difficulty:.2f}")
            return task
        return None
    
    def complete(self,task_id:str,current_tick:int)->Optional[Task]:
        task=self._all.get(task_id)
        if task is None or task.status != TaskStatus.CLAIMED:
            return None
        task.status=TaskStatus.COMPLETED
        task.completed_tick=current_tick
        self.total_completed+=1
        self.log(current_tick,"completed",task_id,task.assigned_agent,f"reward={task.reward_tokens}")
        return task

    def fail(self,task_id:str,current_tick:int)->Optional[Task]:
        task=self._all.get(task_id)
        if task is None or task.status != TaskStatus.CLAIMED:
            return None
        task.status=TaskStatus.FAILED
        self.total_failed+=1
        self.log(current_tick,"failed",task_id,task.assigned_agent,f"penalty={task.penalty_tokens}")
        return task
    
    def pending_count(self)->int:
        return sum(1 for e in self._heap if e.task.status==TaskStatus.PENDING)
    def get(self,task_id:str)->Optional[Task]:
        return self._all.get(task_id)
    
    def recent(self,n:int=5)->list[dict]:
        done=[t for t in self._all.values() if t.status in (TaskStatus.COMPLETED,TaskStatus.FAILED)]
        done.sort(key=lambda t:t.completed_tick or 0 ,reverse=True)
        return [t.to_dict()for t in done[:n]]
    
    def snapshot(self)->dict:
        return{
            "pending": self.pending_count(),
            "completed":self.total_completed,
            "failed":self.total_failed,
            "expired":self.total_expired,
            "posted":self.total_posted,
        }
    def generate_synthetic(self,current_tick:int,n:int=3)->list[Task]:
        """
        Post n synthetic tasks for the 1000-tick stability test.
 
        In Weeks 5+ real tasks replace these. For Week 1, we just
        need tasks flowing through the economy so we can verify that
        token balances diverge.
 
        Difficulty tiers
        ----------------
        easy   0.1–0.35   reward  2– 7   penalty 1–2
        medium 0.36–0.65  reward  8–13   penalty 3–5
        hard   0.66–1.0   reward 14–20   penalty 5–7
        """
        tasks=[]
        for _ in range(n):
            task_type=random.choice(list(TaskType))
            difficulty=random.uniform(0.1,1.0)
            desc= self._synthetic_description(task_type,difficulty)
            task=self.post(task_type,difficulty,desc,current_tick)
            tasks.append(task)
        return tasks

    def _expire(self,current_tick:int):
        for entry in self._heap:
            task=entry.task
            if (task.status == TaskStatus.PENDING and
                    current_tick - task.posted_tick > self.expiry_ticks):
                    task.status=TaskStatus.EXPIRED
                    self.total_expired+=1
                    self.log(current_tick,"expired",task.task_id,None,f"waited {current_tick - task.posted_tick} ticks")
    
    def log(self,tick:int,event:str,task_id:str,agent_id:Optional[int],detail:str):
        self.history.append({
            "tick":tick,
            "event":event,
            "task_id":task_id,
            "agent_id":agent_id,
            "detail" :detail,
        })
    
    @staticmethod
    def _synthetic_description(task_type:TaskType,difficulty:float)->str:
        tier=("easy" if difficulty<0.36 else "medium" if difficulty<0.66 else "hard")

        templates={
            TaskType.CODE:{
                "easy": "write a function to reverse a string",
                "medium" :"implement a binary search on a sorted list",
                "hard" :"build a graph shortest-path algorithm from scratch",

            },
            TaskType.RESEARCH:{
                "easy":"define the concept of overfitting in one paragraph",
                "medium":"compare gradient descent variants with evidence",
                "hard":"propose a falsifiable hypothesis on emergent agnet behaviour",
            },
            TaskType.VISUAL:{
                "easy":"generate a bar chart of token balances",
                "medium":"produce a t-SNE plot of agent fingerprints",
                "hard":"create a force-directed graph of coalition formations",
            },
        }
        return templates[task_type][tier]

if __name__ =="__main__":
    print("===TaskQueue smoke test===\n")
    q=TaskQueue(expiry_ticks=10)
    t1=q.post(TaskType.CODE,0.9,"hard code task",0)
    t2=q.post(TaskType.RESEARCH,0.3,"easy research ",0)
    t3=q.post(TaskType.VISUAL,0.6,"medium visual ",0)

    print(f"Posted 3 tasks. Pending :{q.pending_count()}")
    print(f"t1 reward={t1.reward_tokens} penalty={t1.penalty_tokens}")
    print(f"t2 reward={t2.reward_tokens}  penalty={t2.penalty_tokens}")
    print(f"t3 reward={t3.reward_tokens}  penalty={t3.penalty_tokens}")

    claimed=q.dispatch(agent_id=3,current_tick=1)
    print(f"\nDispatched to agent 3: '{claimed.description}' "
          f"(diff={claimed.difficulty})")
    assert claimed.task_id==t1.task_id,"Priority order token"

    done=q.complete(claimed.task_id,current_tick=2)
    print(f"Completed.Reward :{done.reward_tokens}tokens")

    claimed2=q.dispatch(agent_id=7,current_tick=2)
    print(f"\nDispatched to agent 7: '{claimed2.description}' "
          f"(diff={claimed2.difficulty})")
    assert claimed2.task_id == t3.task_id, "Priority order broken"

    failed = q.fail(claimed2.task_id,current_tick=3)
    print(f"Failed.Penalty {failed.penalty_tokens}tokens")

    remaining =q.dispatch(agent_id=0,current_tick=15)
    print(f"\nDispatch at tick 15 (expiry=10):{remaining}")
    assert remaining is None,"Expiry not working"
    print(f"Expired count:{q.total_expired}")

    print("\n---Synthetic tasks ---")
    synthetic =q.generate_synthetic(current_tick=20,n=5)
    for t in synthetic:
        print(f"  {t.task_type.value:8s}  diff={t.difficulty:.2f}"
              f"  reward={t.reward_tokens:2d}  '{t.description}'")

    print(f"\nSnapshot: {q.snapshot()}")
    print("\n=== all assertions passed ===") 







    


    
