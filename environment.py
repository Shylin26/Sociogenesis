import math
from typing import Dict, Tuple, List, Optional
from logger import logger

class SpatialEnvironment:
    """
    Represents a 2D spatial environment for agents to interact within.
    This allows the simulation to have a concept of 'distance' and 'location',
    which can affect communication latency and resource discovery.
    """
    def __init__(self, width: float = 1000.0, height: float = 1000.0):
        self.width = width
        self.height = height
        # Map agent_id to (x, y) coordinates
        self.agent_positions: Dict[int, Tuple[float, float]] = {}
        # Simple spatial resources: Map resource_id to (x, y, amount)
        self.resources: Dict[str, Tuple[float, float, float]] = {}
        logger.info(f"SpatialEnvironment initialized with dimensions {width}x{height}")

    def add_agent(self, agent_id: int, x: float, y: float):
        """Add an agent to a specific location in the environment."""
        if 0 <= x <= self.width and 0 <= y <= self.height:
            self.agent_positions[agent_id] = (x, y)
            logger.debug(f"Agent {agent_id} placed at ({x:.2f}, {y:.2f})")
        else:
            logger.warning(f"Failed to place Agent {agent_id}: coordinates out of bounds.")

    def move_agent(self, agent_id: int, dx: float, dy: float):
        """Move an agent by a relative amount, bounded by environment size."""
        if agent_id not in self.agent_positions:
            return
        
        x, y = self.agent_positions[agent_id]
        new_x = max(0.0, min(self.width, x + dx))
        new_y = max(0.0, min(self.height, y + dy))
        
        self.agent_positions[agent_id] = (new_x, new_y)

    def get_distance(self, agent1_id: int, agent2_id: int) -> Optional[float]:
        """Calculate Euclidean distance between two agents."""
        if agent1_id not in self.agent_positions or agent2_id not in self.agent_positions:
            return None
        
        x1, y1 = self.agent_positions[agent1_id]
        x2, y2 = self.agent_positions[agent2_id]
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def get_agents_in_radius(self, agent_id: int, radius: float) -> List[int]:
        """Find all other agents within a given radius of a specific agent."""
        if agent_id not in self.agent_positions:
            return []
            
        nearby_agents = []
        for other_id in self.agent_positions:
            if other_id != agent_id:
                dist = self.get_distance(agent_id, other_id)
                if dist is not None and dist <= radius:
                    nearby_agents.append(other_id)
                    
        return nearby_agents
