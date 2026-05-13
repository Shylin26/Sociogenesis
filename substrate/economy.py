class ComputeEconomy:
    def __init__(self,n_agents:int,start_tokens:int=100):
        self.balances={i:start_tokens for i in range(n_agents)}
        self.death_enabled=False # We will enable it later on
        self.history=[]
    def earn(self,agent_id:int,reward:float):
        self.balances[agent_id]+=reward
        self.log(agent_id,'earn',reward)
    def spend(self,agent_id:int,cost:int)->bool:
        if self.balances[agent_id]<cost:
            return False
        self.balances[agent_id]-=cost
        self.log(agent_id,'spend',-cost)
        return True
    def tick(self,agent_id:int):
        """Called once per tick per agent.Checks for death"""
        if self.balances[agent_id]<0 and self.death_enabled:
            return self.kill(agent_id)
        return None
    
    def kill(self,agent_id:int)->dict:
        top=max(self.balances,key=self.balances.get)
        self.balances[agent_id]=self.balances[top]//2
        self.log(agent_id,'death',0)
        return{'killed':agent_id,'parent':top}
    def reward_for_task(self,difficulty:float)->int:
        """difficulty is 0.0–1.0. Returns token reward."""
        return max(1,int(difficulty*10))
    def penalty_for_failure(self,difficulty:float)->int:
        """Failure costs roughly 1/3 of the potential reward."""
        return max(1,int(difficulty*7))
    
    def log(self,agent_id:int,event:str,delta):
        self.history.append({
            'agent':agent_id,
            'event':event,
            'delta':delta,
            'balance':self.balances[agent_id]
        })
    def snapshot(self)->dict:
        return dict(self.balances)



