import random
import uuid
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from coalition.task_decomposer import TaskDecomposer, SubtaskType
from coalition.Auction         import AuctionEngine
from coalition.coalition       import CoalitionFormation, CoalitionStatus
from coalition.aggregator      import CoalitionAggregator

class SimpleRegistry:
    def __init__(self, n):
        self.records = {i: {"agent_id": i, "rep": 0.0, "tokens": 100,
                             "tasks": 0, "alive": True}
                        for i in range(n)}
    def all_living(self): return [v for v in self.records.values() if v["alive"]]
    def update_rep(self, aid, score):
        r = self.records[aid]
        r["rep"] = 0.9 * r["rep"] + 0.1 * score
        r["tasks"] += 1
 
 
class SimpleEconomy:
    def __init__(self, n, start=100):
        self.balances = {i: start for i in range(n)}
    def earn(self, aid, r): self.balances[aid] += r
    def spend(self, aid, c):
        if self.balances[aid] >= c:
            self.balances[aid] -= c
            return True
        return False
    def snapshot(self): return dict(self.balances)

class Agent:
    def __init__(self, agent_id, bias):
        self.agent_id = agent_id
        self.bias = bias
 
    def attempt(self, task_type: str, difficulty: float):
        p       = self.bias.get(task_type, 0.3)
        success = random.random() < p
        quality = random.uniform(0.5, 1.0) * p if success else 0.0
        content = self._make_output(task_type, quality)
        return success, quality, content
 
    def _make_output(self, task_type: str, quality: float) -> str:
        if task_type == "code":
            return (
                f"def solve_{uuid.uuid4().hex[:6]}(data):\n"
                f"    # quality={quality:.2f}\n"
                f"    results = process(data)\n"
                f"    return results\n"
            )
        elif task_type == "research":
            return (
                f"Hypothesis: pattern X dominates in results.\n"
                f"Evidence needed: run solve and count categories.\n"
                f"Experiment: classify each results item by topic.\n"
                f"Falsifiable: if X < 30%, hypothesis rejected.\n"
            )
        else:
            return (
                f"Data flow diagram of solve architecture:\n"
                f"  [Input] → [solve function] → [results]\n"
                f"  → [Classifier] → [Bar Chart of results]\n"
            )

class Week4Society:
 
    TASK_TYPES   = ["code", "research", "visual"]
    HARD_TASK    = (
        "Build a Python web scraper for Hacker News front page. "
        "Write a hypothesis about what topics dominate today. "
        "Generate a data flow diagram of the scraper architecture."
    )
    COALITION_INTERVAL = 10   
 
    def __init__(self, n_agents=10, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
 
        self.n_agents = n_agents
        self.registry = SimpleRegistry(n_agents)
        self.economy  = SimpleEconomy(n_agents)

        gen = torch.Generator(); gen.manual_seed(seed)
        self.type_seeds = {
            name: F.normalize(torch.randn(128, generator=gen), dim=0)
            for name in self.TASK_TYPES
        }

        gen2 = torch.Generator(); gen2.manual_seed(seed + 1)
        self.fingerprints = {}
        for i in range(n_agents):
            base = (self.type_seeds["code"]     if i < 3 else
                    self.type_seeds["research"]  if i < 6 else
                    self.type_seeds["visual"])
            self.fingerprints[i] = F.normalize(
                base + torch.randn(128, generator=gen2) * 0.1, dim=0
            )

        gen3 = torch.Generator(); gen3.manual_seed(seed + 2)
        self.private_latents = {
            i: F.normalize(torch.randn(128, generator=gen3), dim=0)
            for i in range(n_agents)
        }

        self.agents = {}
        for i in range(n_agents):
            if i < 3:
                bias = {"code": 0.82, "research": 0.18, "visual": 0.18}
            elif i < 6:
                bias = {"code": 0.18, "research": 0.82, "visual": 0.18}
            else:
                bias = {"code": 0.18, "research": 0.18, "visual": 0.82}
            self.agents[i] = Agent(i, bias)

        self.decomposer  = TaskDecomposer(min_types=3)
        self.cf          = CoalitionFormation()
        self.aggregator  = CoalitionAggregator()
 
        self.coalition_wins  = 0
        self.coalition_total = 0
        self.solo_wins       = 0
 
        self.tick = 0
    
    def run(self, ticks=200):
        print(f"Running {ticks} ticks, {self.n_agents} agents...")
        print(f"Hard task every {self.COALITION_INTERVAL} ticks\n")
 
        for t in range(ticks):
            self.tick = t
            self._regular_tick(t)
 
            if (t + 1) % self.COALITION_INTERVAL == 0:
                self._coalition_tick(t)
 
            if (t + 1) % 50 == 0:
                self._report(t + 1)
 
        self._final_report(ticks)
    
    def _regular_tick(self, t):
        """Each agent attempts one random typed task."""
        living = [v["agent_id"] for v in self.registry.all_living()]
        for aid in living:
            tt      = random.choice(self.TASK_TYPES)
            diff    = random.uniform(0.3, 0.9)
            reward  = max(1, int(diff * 20))
            penalty = max(1, int(diff * 7))
            success, quality, _ = self.agents[aid].attempt(tt, diff)
            if success:
                self.economy.earn(aid, reward)
                self.registry.update_rep(aid, quality)
            else:
                self.economy.spend(aid, penalty)
    
    def _coalition_tick(self, t):
        
        living     = [v["agent_id"] for v in self.registry.all_living()]
        balances   = self.economy.snapshot()
        reputations= {aid: self.registry.records[aid]["rep"]
                      for aid in range(self.n_agents)}
 
        task_id = str(uuid.uuid4())
        decomp   = self.decomposer.decompose(
            self.HARD_TASK, task_id, difficulty=0.85
        )
        subtasks = decomp.subtasks
        auction_engine = AuctionEngine()
        results = auction_engine.run_all_auctions(
            subtasks       = subtasks,
            living_agents  = living,
            fingerprints   = self.fingerprints,
            token_balances = balances,
            type_seeds     = self.type_seeds,
        )

        for r in results:
            if r.has_winner:
                self.economy.spend(r.winner_id, r.tokens_spent)
        coalition = self.cf.form(
            parent_task_id  = task_id,
            auction_results = results,
            subtasks        = subtasks,
            reputations     = reputations,
            private_latents = self.private_latents,
            token_balances  = self.economy.snapshot(),
            tick            = t,
        )
 
        if coalition is None:
            print(f"  [tick {t}] Coalition formation failed")
            return
        for member in coalition.members:
            success, quality, content = self.agents[member.agent_id].attempt(
                member.subtask_type, 0.85
            )
            self.cf.record_output(
                coalition.coalition_id,
                member.agent_id,
                content,
                quality if success else 0.1,
                t,
            )
        agg_output = self.aggregator.aggregate(coalition)

        completed = self.cf.complete(
            coalition.coalition_id, agg_output.content, t
        )
        for member in completed.members:
            if member.reward_share > 0:
                self.economy.earn(member.agent_id, member.reward_share)
            self.registry.update_rep(member.agent_id, member.quality_score)
        
        solo_qualities = []
        for tt in self.TASK_TYPES:
            best_agent = max(living,
                             key=lambda a: self.agents[a].bias.get(tt, 0))
            _, q, _ = self.agents[best_agent].attempt(tt, 0.85)
            solo_qualities.append(q)
        best_solo = max(solo_qualities)
 
        comparison = self.aggregator.compare_coalition_vs_solo(
            agg_output.quality_score, solo_qualities
        )
 
        self.coalition_total += 1
        if comparison["coalition_wins"]:
            self.coalition_wins += 1
        else:
            self.solo_wins += 1
 
        win_rate = self.coalition_wins / self.coalition_total
        print(f"  [tick {t+1:3d}] coalition={agg_output.quality_score:.3f}  "
              f"solo={comparison['best_solo_quality']:.3f}  "
              f"{'COALITION' if comparison['coalition_wins'] else '✗ SOLO'}  "
              f"win_rate={win_rate:.0%}  "
              f"bonus={agg_output.cross_ref_bonus:.2f}")
        
    def _report(self, t):
        balances = self.economy.snapshot()
        win_rate = (self.coalition_wins / self.coalition_total
                    if self.coalition_total > 0 else 0)
        print(f"\n── tick {t:4d} ──────────────────────────────")
        print(f"  coalition win rate : {win_rate:.0%} "
              f"({self.coalition_wins}/{self.coalition_total})")
        print(f"  token range        : "
              f"min={min(balances.values())}  "
              f"max={max(balances.values())}")
        print(f"  coalitions formed  : {self.cf.total_formed}")
        print()
 
    def _final_report(self, ticks):
        print("═" * 52)
        print("WEEK 4 FINAL REPORT")
        print("═" * 52)
 
        balances = self.economy.snapshot()
        win_rate = (self.coalition_wins / self.coalition_total
                    if self.coalition_total > 0 else 0)
 
        c1 = self.cf.total_formed >= 5
        c2 = win_rate >= 0.60
        c3 = self.cf.total_failed == 0
 
        print(f"\n{'✓' if c1 else '✗'} coalitions formed   : "
              f"{self.cf.total_formed}")
        print(f"{'✓' if c2 else '✗'} coalition win rate  : "
              f"{win_rate:.0%}  (need ≥60%)")
        print(f"{'✓' if c3 else '✗'} zero formation failures")
 
        print(f"\nCoalition breakdown:")
        print(f"  total hard tasks  : {self.coalition_total}")
        print(f"  coalition wins    : {self.coalition_wins}")
        print(f"  solo wins         : {self.solo_wins}")
 
        print(f"\nFinal token balances:")
        for aid, bal in sorted(balances.items()):
            tag = ("CODE" if aid < 3 else
                   "RES " if aid < 6 else "VIS ")
            bar = "█" * max(0, bal // 20)
            print(f"  A{aid} [{tag}]: {bal:5d}  {bar}")
 
        recent = self.cf.recent_completed(3)
        print(f"\nRecent coalitions:")
        for c in recent:
            print(f"  {c['coalition_id'][:12]}... "
                  f"members={[m['agent_id'] for m in c['members']]}  "
                  f"coord={c['coordinator_id']}")
 
        all_pass = c1 and c2
        print(f"\n{' WEEK 4 COMPLETE' if all_pass else 'Some criteria not met'}")
        print("═" * 52)
 

if __name__ == "__main__":
    print("=" * 52)
    print("PANTHEON Week 4 — Coalition Formation Loop")
    print("=" * 52 + "\n")
 
    society = Week4Society(n_agents=10, seed=42)
    society.run(ticks=200)
 





