import uuid
from dataclasses import dataclass, field
from typing import Optional
 
import torch
import torch.nn.functional as F

@dataclass
class Bid:
    agent_id    : int
    subtask_id  : str
    skill_match : float
    bid_cost    : int
    bid_score   : float
    can_afford  : bool
    def to_dict(self) -> dict:
        return {
            "agent_id"    : self.agent_id,
            "subtask_id"  : self.subtask_id,
            "skill_match" : round(self.skill_match, 3),
            "bid_cost"    : self.bid_cost,
            "bid_score"   : round(self.bid_score, 3),
            "can_afford"  : self.can_afford,
        }

 
@dataclass
class AuctionResult:
    subtask_id   : str
    winner_id    : Optional[int]
    winner_bid   : Optional[Bid]
    all_bids     : list[Bid]
    tokens_spent : int = 0
 
    @property
    def has_winner(self) -> bool:
        return self.winner_id is not None
 
    def to_dict(self) -> dict:
        return {
            "subtask_id"   : self.subtask_id,
            "winner_id"    : self.winner_id,
            "tokens_spent" : self.tokens_spent,
            "n_bidders"    : len(self.all_bids),
            "top_bids"     : [b.to_dict() for b in
                              sorted(self.all_bids,
                                     key=lambda b: b.bid_score,
                                     reverse=True)[:3]],
        }

class AuctionEngine:
    def __init__(self, min_coalition : int = 2,
                 max_coalition : int = 7,
                 min_balance   : int = 5):
        self.min_coalition = min_coalition
        self.max_coalition = max_coalition
        self.min_balance   = min_balance
        self.history : list[dict] = []
        self.total_auctions_run = 0
        self.total_tokens_spent = 0

    def run_auction(self, subtask,
                    living_agents  : list[int],
                    fingerprints   : dict[int, torch.Tensor],
                    token_balances : dict[int, int],
                    task_embedding : torch.Tensor) -> AuctionResult:
        bids = self._collect_bids(
            subtask        = subtask,
            living_agents  = living_agents,
            fingerprints   = fingerprints,
            token_balances = token_balances,
            task_embedding = task_embedding,
        )
 
        result = self._select_winner(subtask.subtask_id, bids)
        self.total_auctions_run += 1
        self.total_tokens_spent += result.tokens_spent
        self.history.append({
            "subtask_id" : subtask.subtask_id,
            "task_type"  : subtask.task_type.value,
            "result"     : result.to_dict(),
        })
 
        return result

    def run_all_auctions(self, subtasks      : list,
                         living_agents  : list[int],
                         fingerprints   : dict[int, torch.Tensor],
                         token_balances : dict[int, int],
                         type_seeds     : dict[str, torch.Tensor],
                         encoder        = None) -> list[AuctionResult]:
        results       = []
        winners_so_far = set()
        for subtask in subtasks:
            if len(winners_so_far) >= self.max_coalition:
                break
            eligible = [a for a in living_agents
                        if a not in winners_so_far]
            if not eligible:
                break
            if encoder is not None:
                with torch.no_grad():
                    task_emb = encoder(subtask.description)
            else:
                task_emb = type_seeds.get(
                    subtask.task_type.value,
                    torch.randn(128)
                )
                task_emb = F.normalize(task_emb, dim=0)
 
            result = self.run_auction(
                subtask        = subtask,
                living_agents  = eligible,
                fingerprints   = fingerprints,
                token_balances = token_balances,
                task_embedding = task_emb,
            )
 
            results.append(result)
            if result.has_winner:
                winners_so_far.add(result.winner_id)
 
        return results


    def _collect_bids(self, subtask,
                      living_agents  : list[int],
                      fingerprints   : dict[int, torch.Tensor],
                      token_balances : dict[int, int],
                      task_embedding : torch.Tensor) -> list[Bid]:

        bids = []
        task_emb = F.normalize(task_embedding.float(), dim=0)
 
        for agent_id in living_agents:
            balance = token_balances.get(agent_id, 0)
            if balance < self.min_balance:
                continue   # too poor to bid
 
            fp = fingerprints.get(agent_id)
            if fp is None:
                continue
 
            fp          = F.normalize(fp.float(), dim=0)
            skill_match = torch.dot(task_emb, fp).item()
            bid_cost    = max(1, int(10 * (1 - skill_match)))
            bid_score   = skill_match - bid_cost / 20.0
            can_afford  = balance >= bid_cost
 
            bids.append(Bid(
                agent_id    = agent_id,
                subtask_id  = subtask.subtask_id,
                skill_match = skill_match,
                bid_cost    = bid_cost,
                bid_score   = bid_score,
                can_afford  = can_afford,
            ))
 
        return bids
    
    def _select_winner(self, subtask_id: str,
                       bids: list[Bid]) -> AuctionResult:
        affordable = [b for b in bids if b.can_afford]
 
        if not affordable:
            return AuctionResult(
                subtask_id   = subtask_id,
                winner_id    = None,
                winner_bid   = None,
                all_bids     = bids,
                tokens_spent = 0,
            )
 
        winner = max(affordable, key=lambda b: b.bid_score)
 
        return AuctionResult(
            subtask_id   = subtask_id,
            winner_id    = winner.agent_id,
            winner_bid   = winner,
            all_bids     = bids,
            tokens_spent = winner.bid_cost,
        )
 
    def snapshot(self) -> dict:
        return {
            "total_auctions" : self.total_auctions_run,
            "total_spent"    : self.total_tokens_spent,
            "history_len"    : len(self.history),
        }

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from coalition.task_decomposer import TaskDecomposer, SubtaskType
    import torch.nn.functional as F
 
    print("=" * 56)
    print("PANTHEON Week 4 — AuctionEngine smoke test")
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
        if i < 3:
            base = type_seeds["code"]
        elif i < 6:
            base = type_seeds["research"]
        else:
            base = type_seeds["visual"]
        noise = torch.randn(128, generator=gen2) * 0.15
        fingerprints[i] = F.normalize(base + noise, dim=0)
    
    balances = {i: 100 for i in range(N_AGENTS)}

    decomposer = TaskDecomposer(min_types=3)
    result     = decomposer.decompose_demo_task()
    subtasks   = result.subtasks

    print(f"\nDecomposed into {len(subtasks)} subtasks:")
    for s in subtasks:
        print(f"  [{s.task_type.value:8s}] diff={s.difficulty:.2f} "
              f"reward={s.reward}")
    
    print("\n── Test 1: single auction (code subtask) ──")
    engine   = AuctionEngine(min_coalition=2, max_coalition=7)
    code_sub = next(s for s in subtasks if s.task_type == SubtaskType.CODE)
    task_emb = type_seeds["code"]
 
    ar = engine.run_auction(
        subtask        = code_sub,
        living_agents  = list(range(N_AGENTS)),
        fingerprints   = fingerprints,
        token_balances = balances,
        task_embedding = task_emb,
    )
 
    print(f"  winner    : agent {ar.winner_id}")
    print(f"  tokens_spent: {ar.tokens_spent}")
    print(f"  top 3 bids:")
    for b in sorted(ar.all_bids, key=lambda b: b.bid_score, reverse=True)[:3]:
        print(f"    agent {b.agent_id}: "
              f"match={b.skill_match:.3f} "
              f"cost={b.bid_cost} "
              f"score={b.bid_score:.3f}")

    assert ar.winner_id is not None, "No winner found"
    assert ar.winner_id < 3, \
        f"Expected code specialist (0-2), got agent {ar.winner_id}"
    print(f" code subtask won by code specialist (agent {ar.winner_id})")

    print("\n── Test 2: full coalition auction ──")
    engine2  = AuctionEngine(min_coalition=2, max_coalition=7)
    results  = engine2.run_all_auctions(
        subtasks       = subtasks,
        living_agents  = list(range(N_AGENTS)),
        fingerprints   = fingerprints,
        token_balances = balances,
        type_seeds     = type_seeds,
        encoder        = None, 
    )
 
    print(f"  auctions run : {len(results)}")
    for r in results:
        sub = next(s for s in subtasks if s.subtask_id == r.subtask_id)
        print(f"  [{sub.task_type.value:8s}] "
              f"winner=agent {r.winner_id}  "
              f"spent={r.tokens_spent}  "
              f"bidders={len(r.all_bids)}")
 
    winners = [r.winner_id for r in results if r.has_winner]
    assert len(winners) >= 2, "Coalition too small"
    assert len(set(winners)) == len(winners), "Duplicate winners"
    print(f" coalition formed: {len(winners)} unique winners")

    for r in results:
        sub = next(s for s in subtasks if s.subtask_id == r.subtask_id)
        if r.winner_id is not None:
            tt = sub.task_type.value
            expected = {"code": range(3), "research": range(3,6),
                        "visual": range(6,10)}
            if tt in expected:
                assert r.winner_id in expected[tt], \
                    f"{tt} won by agent {r.winner_id} — not a specialist"
    print("each subtask won by correct specialist")
    print("\n── Test 3: poor agents cannot bid ──")
    poor_balances = {i: 2 for i in range(N_AGENTS)}   
    engine3 = AuctionEngine(min_balance=5)
    ar3 = engine3.run_auction(
        subtask        = code_sub,
        living_agents  = list(range(N_AGENTS)),
        fingerprints   = fingerprints,
        token_balances = poor_balances,
        task_embedding = task_emb,
    )
    assert ar3.winner_id is None, "Poor agent should not win"
    print(" no winner when all agents too poor")

    print("\n── Test 4: max coalition cap (max=2) ──")

    extra_tasks = subtasks * 2   
    engine4 = AuctionEngine(max_coalition=2)
    results4 = engine4.run_all_auctions(
        subtasks       = extra_tasks[:5],
        living_agents  = list(range(N_AGENTS)),
        fingerprints   = fingerprints,
        token_balances = balances,
        type_seeds     = type_seeds,
    )
    winners4 = [r.winner_id for r in results4 if r.has_winner]
    assert len(set(winners4)) <= 2, "Coalition exceeded max size"
    print(f" coalition capped at max=2: {len(set(winners4))} unique winners")

    print(f"\nSnapshot: {engine2.snapshot()}")
 
    print("\n" + "=" * 56)
    print("AuctionEngine — DONE")
    print("Next: coalition/coalition.py")
    print("=" * 56)












    



 
