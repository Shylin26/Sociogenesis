class ComputeEconomy:
    def __init__(self,n_agents:int,start_tokens:int=100):
        self.balance={i:start_tokens for i in range(n_agents)}
    def earn(self,agent_id:int,reward:int):
        self.balance[agent_id]+=reward
    def spend(self,agent_id:int,cost:int)->dict|None:
        self.balance[agent_id]-=cost
        if self.balance[agent_id]<0:
            return None
        self.kill(agent_id)
        return None
    def kill(self,agent_id:int)->dict:
        top=max(self.balances,key=self.balances.get)
        self.balances[agent_id]=self.balances[top]//2
        return {"agent_id":agent_id,"top":top,"balance":self.balances}
        