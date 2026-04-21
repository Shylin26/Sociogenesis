import asyncio
import asyncio
from economy import ComputeEconomy
from registery import AgentRegistery
from shared_memory import SharedMemory
from communication_bus import CommunicationBus
from task_queue import TaskQueue
from agent import BaseAgent
async def main():
    n_agents=10
    max_ticks=1000
    print(f"Initialising World with {n_agents} agents...")
    economy=ComputeEconomy(n_agents=n_agents,start_tokens=100)
    registery=AgentRegistery(n_agents=n_agents)
    memory=SharedMemory()
    bus=CommunicationBus()
    tasks=TaskQueue()
    agents=[]
    for i in range(n_agents):
        agent=BaseAgent(i,economy,registery,memory,bus,tasks)
        agents.append(agent)
        
    print(f"Starting {max_ticks} ticks of simulation...")
    agent_tasks=[asyncio.create_task(agent.run(max_ticks)) for agent in agents]
    await asyncio.gather(*agent_tasks)
    
    print("\n--- Simulation Over ---")
    print("Final Economy Check (Did wealth diverge?):")
    for i in range(n_agents):
        print(f"Agent {i:02d}: {economy.balances[i]} tokens")

if __name__=="__main__":
    asyncio.run(main())




