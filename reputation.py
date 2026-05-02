import time
from typing import Dict, List, Optional
from logger import logger

class ReputationSystem:
    """
    Manages the reputation scores of agents within the Sociogenesis simulation.
    Reputation affects an agent's ability to participate in the economy, 
    acquire tasks, and influence governance.
    """
    def __init__(self, n_agents: int):
        # Initialize all agents with a neutral reputation score of 50.0
        self.scores: Dict[int, float] = {i: 50.0 for i in range(n_agents)}
        self.history: Dict[int, List[float]] = {i: [50.0] for i in range(n_agents)}
        logger.info(f"ReputationSystem initialized for {n_agents} agents.")

    def update_reputation(self, agent_id: int, delta: float, reason: str = ""):
        """Update reputation score for an agent, bounded between 0.0 and 100.0."""
        if agent_id not in self.scores:
            logger.warning(f"Attempted to update reputation for unknown agent {agent_id}")
            return
        
        old_score = self.scores[agent_id]
        new_score = max(0.0, min(100.0, old_score + delta))
        self.scores[agent_id] = new_score
        self.history[agent_id].append(new_score)
        
        log_msg = f"Agent {agent_id} reputation changed: {old_score:.2f} -> {new_score:.2f} (delta: {delta:+.2f})"
        if reason:
            log_msg += f" Reason: {reason}"
        logger.debug(log_msg)

    def get_score(self, agent_id: int) -> float:
        """Retrieve the current reputation score for a specific agent."""
        return self.scores.get(agent_id, 0.0)

    def get_top_agents(self, limit: int = 5) -> List[int]:
        """Returns a list of agent IDs sorted by highest reputation."""
        sorted_agents = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        return [agent_id for agent_id, score in sorted_agents[:limit]]
        
    def penalize_bad_actor(self, agent_id: int, penalty: float = 5.0):
        """Apply a penalty to an agent for malicious or faulty behavior."""
        self.update_reputation(agent_id, -abs(penalty), reason="Bad actor penalty")
        
    def reward_good_actor(self, agent_id: int, reward: float = 2.0):
        """Apply a reward to an agent for positive contributions to the simulation."""
        self.update_reputation(agent_id, abs(reward), reason="Good actor reward")
