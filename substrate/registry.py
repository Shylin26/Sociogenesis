import torch
from dataclasses import dataclass, field
@dataclass
class AgentRecord:
    agent_id:int
    skill_fingerprint:torch.Tensor
    reputation_score:float=0.0
    token_balance:int=100
    parent_id: int | None = None
    birth_tick:int=0
    task_completed:int=0
class AgentRegistry:
    def __init__(self,n_agents:int):
        self.records:dict[int,AgentRecord]={
            i:AgentRecord(
                agent_id=i,
                skill_fingerprint=torch.zeros(128)

            )
            for i in range(n_agents)
        }  
    def get(self,agent_id:int)->AgentRecord:
        return self.records[agent_id]
    
    def all_living(self)->list[AgentRecord]:
        return list(self.records.values())
    def update_fingerprint(self,agent_id:int,new_fp:torch.Tensor):
        self.records[agent_id].skill_fingerprint=new_fp/new_fp.norm()
    def update_reputation(self,agent_id:int,score:float):
        r=self.records[agent_id]
        r.reputation_score=0.9*r.reputation_score+0.1*score
    def replace_agent(self,dead_id:int,parent_id:int,current_tick:int,mutation_rate:float=0.1):
        parent=self.records[parent_id]
        new_fp=parent.skill_fingerprint+mutation_rate*torch.randn(128)
        self.records[dead_id]=AgentRecord(
            agent_id=dead_id,
            skill_fingerprint=new_fp/new_fp.norm(),
            token_balance=parent.token_balance//2,
            parent_id=parent_id,
            birth_tick=current_tick
        )
        