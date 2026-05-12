from typing import Dict, Set, Optional, List, Tuple

class SocialNetwork:
    """
    Models the social graph of connections between agents.
    Tracks relationships, trust links, and network topology
    for the Sociogenesis multi-agent system.
    """
    def __init__(self):
        self._edges: Dict[str, Set[str]] = {}
        self._weights: Dict[Tuple[str, str], float] = {}

    def add_agent(self, agent_id: str):
        """Registers a new agent node in the network."""
        if agent_id not in self._edges:
            self._edges[agent_id] = set()

    def connect(self, agent_a: str, agent_b: str, weight: float = 1.0):
        """Creates a bidirectional link between two agents."""
        for agent in (agent_a, agent_b):
            if agent not in self._edges:
                self.add_agent(agent)

        self._edges[agent_a].add(agent_b)
        self._edges[agent_b].add(agent_a)
        self._weights[(agent_a, agent_b)] = weight
        self._weights[(agent_b, agent_a)] = weight

    def disconnect(self, agent_a: str, agent_b: str):
        """Removes the link between two agents."""
        self._edges.get(agent_a, set()).discard(agent_b)
        self._edges.get(agent_b, set()).discard(agent_a)
        self._weights.pop((agent_a, agent_b), None)
        self._weights.pop((agent_b, agent_a), None)

    def get_neighbors(self, agent_id: str) -> Set[str]:
        """Returns the set of agents directly connected to the given agent."""
        return set(self._edges.get(agent_id, set()))

    def get_weight(self, agent_a: str, agent_b: str) -> Optional[float]:
        """Returns the connection weight between two agents, if connected."""
        return self._weights.get((agent_a, agent_b))

    def degree(self, agent_id: str) -> int:
        """Returns the number of direct connections an agent has."""
        return len(self._edges.get(agent_id, set()))

    def all_agents(self) -> List[str]:
        """Returns a list of all agents in the network."""
        return list(self._edges.keys())

    def is_connected(self, agent_a: str, agent_b: str) -> bool:
        """Checks whether two agents are directly connected."""
        return agent_b in self._edges.get(agent_a, set())
