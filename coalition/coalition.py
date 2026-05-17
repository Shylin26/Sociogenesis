import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch


class CoalitionStatus(Enum):
    FORMING   = "forming"
    ACTIVE    = "active"
    MERGING   = "merging"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class CoalitionMember:
    agent_id      : int
    subtask_id    : str
    subtask_type  : str
    bid_cost_paid : int
    reward_share  : int   = 0
    output        : str   = ""
    quality_score : float = 0.0
    completed     : bool  = False


@dataclass
class Coalition:
    coalition_id    : str
    parent_task_id  : str
    members         : list[CoalitionMember]
    coordinator_id  : int
    status          : CoalitionStatus = CoalitionStatus.FORMING
    token_pool      : int             = 0
    formed_tick     : int             = 0
    completed_tick  : Optional[int]   = None
    final_output    : str             = ""
    _member_latents : dict            = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "coalition_id"   : self.coalition_id,
            "parent_task_id" : self.parent_task_id,
            "coordinator_id" : self.coordinator_id,
            "status"         : self.status.value,
            "token_pool"     : self.token_pool,
            "n_members"      : len(self.members),
            "members"        : [
                {"agent_id"     : m.agent_id,
                 "subtask_type" : m.subtask_type,
                 "completed"    : m.completed,
                 "quality"      : round(m.quality_score, 3)}
                for m in self.members
            ],
            "formed_tick"    : self.formed_tick,
            "completed_tick" : self.completed_tick,
        }

    @property
    def member_ids(self) -> list[int]:
        return [m.agent_id for m in self.members]

    @property
    def is_complete(self) -> bool:
        return all(m.completed for m in self.members)

    @property
    def mean_quality(self) -> float:
        scores = [m.quality_score for m in self.members if m.completed]
        return sum(scores) / len(scores) if scores else 0.0

    def get_member(self, agent_id: int) -> Optional[CoalitionMember]:
        for m in self.members:
            if m.agent_id == agent_id:
                return m
        return None


class CoalitionFormation:
    def __init__(self):
        self.active_coalitions    : dict[str, Coalition] = {}
        self.completed_coalitions : list[Coalition]      = []
        self.total_formed         : int = 0
        self.total_completed      : int = 0
        self.total_failed         : int = 0

    def form(self,
             parent_task_id  : str,
             auction_results : list,
             subtasks        : list,
             reputations     : dict[int, float],
             private_latents : dict[int, torch.Tensor],
             token_balances  : dict[int, int],
             tick            : int) -> Optional[Coalition]:

        winners = [(r, s) for r, s in zip(auction_results, subtasks)
                   if r.has_winner]

        if len(winners) < 2:
            self.total_failed += 1
            return None

        members = []
        for auction_result, subtask in winners:
            members.append(CoalitionMember(
                agent_id      = auction_result.winner_id,
                subtask_id    = subtask.subtask_id,
                subtask_type  = subtask.task_type.value,
                bid_cost_paid = auction_result.tokens_spent,
            ))

        coordinator_id = max(
            [m.agent_id for m in members],
            key=lambda aid: reputations.get(aid, 0.0)
        )

        token_pool = sum(s.reward for _, s in winners)

        latent_snapshot = {}
        for m in members:
            lat = private_latents.get(m.agent_id)
            if lat is not None:
                latent_snapshot[m.agent_id] = lat.detach().clone()

        coalition = Coalition(
            coalition_id    = str(uuid.uuid4()),
            parent_task_id  = parent_task_id,
            members         = members,
            coordinator_id  = coordinator_id,
            status          = CoalitionStatus.ACTIVE,
            token_pool      = token_pool,
            formed_tick     = tick,
            _member_latents = latent_snapshot,
        )

        self.active_coalitions[coalition.coalition_id] = coalition
        self.total_formed += 1
        return coalition

    def get_coordinator_directive(self, coalition: Coalition,
                                  target_agent_id: int) -> torch.Tensor:
        coord_latent  = coalition._member_latents.get(coalition.coordinator_id)
        member_latent = coalition._member_latents.get(target_agent_id)

        if coord_latent is None or member_latent is None:
            return torch.zeros(128)

        alignment = torch.dot(coord_latent, member_latent).item()
        directive = alignment * member_latent + (1 - abs(alignment)) * coord_latent

        norm = directive.norm()
        if norm > 1e-8:
            directive = directive / norm

        return directive

    def get_all_directives(self, coalition: Coalition) -> dict[int, torch.Tensor]:
        return {
            m.agent_id: self.get_coordinator_directive(coalition, m.agent_id)
            for m in coalition.members
            if m.agent_id != coalition.coordinator_id
        }

    def record_output(self, coalition_id : str,
                      agent_id     : int,
                      output       : str,
                      quality      : float,
                      tick         : int):
        coalition = self.active_coalitions.get(coalition_id)
        if coalition is None:
            return

        member = coalition.get_member(agent_id)
        if member is None:
            return

        member.output        = output
        member.quality_score = quality
        member.completed     = True

    def complete(self, coalition_id : str,
                 final_output : str,
                 tick         : int) -> Optional[Coalition]:
        coalition = self.active_coalitions.get(coalition_id)
        if coalition is None:
            return None

        coalition.status         = CoalitionStatus.COMPLETED
        coalition.final_output   = final_output
        coalition.completed_tick = tick

        total_quality = sum(m.quality_score for m in coalition.members)
        if total_quality > 0:
            for member in coalition.members:
                share = int(
                    coalition.token_pool *
                    (member.quality_score / total_quality)
                )
                member.reward_share = share
        else:
            per_member = coalition.token_pool // len(coalition.members)
            for member in coalition.members:
                member.reward_share = per_member

        self.completed_coalitions.append(coalition)
        del self.active_coalitions[coalition_id]
        self.total_completed += 1

        return coalition

    def fail(self, coalition_id: str, tick: int):
        coalition = self.active_coalitions.get(coalition_id)
        if coalition:
            coalition.status         = CoalitionStatus.FAILED
            coalition.completed_tick = tick
            self.completed_coalitions.append(coalition)
            del self.active_coalitions[coalition_id]
            self.total_failed += 1

    def get_active(self) -> list[Coalition]:
        return list(self.active_coalitions.values())

    def recent_completed(self, n: int = 5) -> list[dict]:
        return [c.to_dict() for c in self.completed_coalitions[-n:]]

    def snapshot(self) -> dict:
        return {
            "active"    : len(self.active_coalitions),
            "completed" : self.total_completed,
            "failed"    : self.total_failed,
            "formed"    : self.total_formed,
        }


if __name__ == "__main__":
    from coalition.task_decomposer import TaskDecomposer, SubtaskType
    from coalition.Auction import AuctionEngine, AuctionResult
    import torch.nn.functional as F

    print("=" * 56)
    print("PANTHEON Week 4 — Coalition smoke test")
    print("=" * 56)

    N_AGENTS   = 10
    TASK_TYPES = ["code", "research", "visual"]
    gen = torch.Generator(); gen.manual_seed(42)
    type_seeds = {}
    for name in TASK_TYPES:
        type_seeds[name] = F.normalize(torch.randn(128, generator=gen), dim=0)

    fingerprints = {}
    gen2 = torch.Generator(); gen2.manual_seed(7)
    for i in range(N_AGENTS):
        base = (type_seeds["code"] if i < 3 else
                type_seeds["research"] if i < 6 else
                type_seeds["visual"])
        fingerprints[i] = F.normalize(
            base + torch.randn(128, generator=gen2) * 0.15, dim=0
        )

    balances    = {i: 100 for i in range(N_AGENTS)}
    reputations = {i: float(i) * 0.1 for i in range(N_AGENTS)}

    gen3 = torch.Generator(); gen3.manual_seed(99)
    private_latents = {
        i: F.normalize(torch.randn(128, generator=gen3), dim=0)
        for i in range(N_AGENTS)
    }

    decomposer = TaskDecomposer(min_types=3)
    decomp     = decomposer.decompose_demo_task()
    subtasks   = decomp.subtasks

    auction_engine = AuctionEngine()
    results = auction_engine.run_all_auctions(
        subtasks       = subtasks,
        living_agents  = list(range(N_AGENTS)),
        fingerprints   = fingerprints,
        token_balances = balances,
        type_seeds     = type_seeds,
    )

    print("\n── Test 1: form coalition ──")
    cf = CoalitionFormation()
    coalition = cf.form(
        parent_task_id  = decomp.task_id,
        auction_results = results,
        subtasks        = subtasks,
        reputations     = reputations,
        private_latents = private_latents,
        token_balances  = balances,
        tick            = 42,
    )

    assert coalition is not None
    print(f"  coalition_id  : {coalition.coalition_id[:12]}...")
    print(f"  members       : {coalition.member_ids}")
    print(f"  coordinator   : agent {coalition.coordinator_id}")
    print(f"  token_pool    : {coalition.token_pool}")
    print(f"  status        : {coalition.status.value}")
    assert len(coalition.members) >= 2
    assert coalition.status == CoalitionStatus.ACTIVE
    assert coalition.coordinator_id == max(
        coalition.member_ids, key=lambda a: reputations[a]
    )
    print("coalition formed correctly")

    print("\n── Test 2: coordinator directives ──")
    directives = cf.get_all_directives(coalition)
    for aid, directive in directives.items():
        print(f"  directive for agent {aid}: "
              f"norm={directive.norm().item():.3f}")
        assert directive.shape == (128,)
        assert directive.norm().item() > 0
    print("directives computed for all non-coordinator members")

    print("\n── Test 3: record member outputs ──")
    for member in coalition.members:
        cf.record_output(
            coalition_id = coalition.coalition_id,
            agent_id     = member.agent_id,
            output       = f"output from agent {member.agent_id}",
            quality      = 0.5 + member.agent_id * 0.04,
            tick         = 50,
        )
    assert coalition.is_complete
    print(f"  mean quality: {coalition.mean_quality:.3f}")
    print("all members completed")

    print("\n── Test 4: complete coalition + reward split ──")
    completed = cf.complete(
        coalition_id = coalition.coalition_id,
        final_output = "merged final output",
        tick         = 55,
    )
    assert completed.status == CoalitionStatus.COMPLETED
    total_rewarded = sum(m.reward_share for m in completed.members)
    print(f"  token_pool    : {completed.token_pool}")
    print(f"  total rewarded: {total_rewarded}")
    print(f"  reward shares : "
          f"{[(m.agent_id, m.reward_share) for m in completed.members]}")
    assert total_rewarded <= completed.token_pool
    print("reward distributed proportional to quality")

    print("\n── Test 5: coalition fails with 1 winner ──")
    fake_results = [AuctionResult(
        subtask_id="x", winner_id=0, winner_bid=None,
        all_bids=[], tokens_spent=5
    )]
    fake_subtasks = [subtasks[0]]
    c2 = cf.form("task-x", fake_results, fake_subtasks,
                 reputations, private_latents, balances, tick=60)
    assert c2 is None
    print("coalition not formed with only 1 winner")

    print(f"\nSnapshot: {cf.snapshot()}")
    print("\n" + "=" * 56)
    print("Coalition — DONE")
    print("Next: coalition/aggregator.py")
    print("=" * 56)
