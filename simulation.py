import asyncio
import asyncio
from economy import ComputeEconomy
from registery import AgentRegistery
from shared_memory import SharedMemory
from communication_bus import CommunicationBus
from task_queue import TaskQueue
from agent import BaseAgent
from metrics import WealthMetrics
from governance import Governance

async def main():
    n_agents=10
    max_ticks=1000
    print(f"Initialising World with {n_agents} agents...")
    economy=ComputeEconomy(n_agents=n_agents,start_tokens=100)
    gov = Governance(economy)
    registery=AgentRegistery(n_agents=n_agents)
    memory=SharedMemory()
    bus=CommunicationBus()
    tasks=TaskQueue()
    agents=[]
    for i in range(n_agents):
        agent=BaseAgent(i,economy,registery,memory,bus,tasks)
        agents.append(agent)
        
    print(f"Starting {max_ticks} ticks of simulation with Governance...")
    
    # We run the simulation in chunks to allow governance interventions
    chunk_size = 100
    for chunk in range(0, max_ticks, chunk_size):
        agent_tasks=[asyncio.create_task(agent.run(chunk_size)) for agent in agents]
        await asyncio.gather(*agent_tasks)
        
        # Governance intervention
        collected = gov.collect_taxes()
        gov.redistribute_wealth(strategy="equal")
        print(f"[Tick {chunk + chunk_size}] Governance: Taxed {collected} tokens and redistributed.")
    
    print("\n--- Simulation Over ---")
    metrics = WealthMetrics.get_summary(economy.balances)
    print("\nEconomy Analysis:")
    print(f"Total Wealth: {metrics['total_wealth']}")
    print(f"Mean Balance: {metrics['mean']:.2f}")
    print(f"Median Balance: {metrics['median']:.2f}")
    print(f"Standard Deviation: {metrics['std_dev']:.2f}")
    print(f"Max Balance: {metrics['max']}")
    print(f"Min Balance: {metrics['min']}")
    print(f"Gini Coefficient: {metrics['gini']:.4f}")
    
    print("\nFinal Agent Balances:")
    for i in range(n_agents):
        print(f"Agent {i:02d}: {economy.balances[i]} tokens")

if __name__=="__main__":
    asyncio.run(main())




