from typing import List, Dict, Any, Optional

class ConsensusMechanism:
    """
    Implements basic consensus algorithms for agents to reach agreement
    on shared state or decisions within the Sociogenesis environment.
    """
    def __init__(self, required_majority: float = 0.51):
        self.required_majority = required_majority
        self.active_proposals: Dict[str, Dict[str, Any]] = {}

    def create_proposal(self, proposal_id: str, creator_id: str, data: Any) -> bool:
        """Initiates a new proposal for agents to vote on."""
        if proposal_id in self.active_proposals:
            return False
            
        self.active_proposals[proposal_id] = {
            "creator": creator_id,
            "data": data,
            "votes": {"yes": set(), "no": set()},
            "status": "pending"
        }
        return True

    def cast_vote(self, proposal_id: str, agent_id: str, approve: bool) -> bool:
        """Allows an agent to cast a vote on an active proposal."""
        if proposal_id not in self.active_proposals:
            return False
            
        proposal = self.active_proposals[proposal_id]
        if proposal["status"] != "pending":
            return False
            
        # Remove previous vote if any
        proposal["votes"]["yes"].discard(agent_id)
        proposal["votes"]["no"].discard(agent_id)
        
        if approve:
            proposal["votes"]["yes"].add(agent_id)
        else:
            proposal["votes"]["no"].add(agent_id)
            
        return True

    def evaluate_proposal(self, proposal_id: str, total_eligible_voters: int) -> Optional[bool]:
        """
        Evaluates if a proposal has reached the required majority.
        Returns True if passed, False if rejected, None if still pending.
        """
        if proposal_id not in self.active_proposals:
            return None
            
        proposal = self.active_proposals[proposal_id]
        if proposal["status"] != "pending":
            return proposal["status"] == "passed"
            
        yes_votes = len(proposal["votes"]["yes"])
        no_votes = len(proposal["votes"]["no"])
        
        # Check if mathematically impossible to reach majority
        if yes_votes >= total_eligible_voters * self.required_majority:
            proposal["status"] = "passed"
            return True
        elif no_votes > total_eligible_voters * (1 - self.required_majority):
            proposal["status"] = "rejected"
            return False
            
        return None
