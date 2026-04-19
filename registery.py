import torch
class AgentRegistery:
    def __init__(self,n_agents:int):
        self.agents={}
        for i in range(n_agents):
            self.agents[i] = {
                'id': i,
                'reputation': 1.0,
                'fingerprint': torch.zeros(128)
            }
    def get_agent(self,agent_id:int):
        return self.agents[agent_id]
    def update_reputation(self,agent_id:int,change:float):
        self.agents[agent_id]['reputation']+=change
        self.agents[agent_id]['reputation']=max(0.1,min(10.0,self.agents[agent_id]['reputation']))
    def active_agents(self):
        return list(self.agents.keys())