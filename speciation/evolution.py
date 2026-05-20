import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field

from speciation.engine import SpeciationEngine
from substrate.economy import ComputeEconomy
from substrate.registry import AgentRegistry


EVOLUTION_INTERVAL  = 200
BOTTOM_PERCENTILE   = 0.10
TOP_PERCENTILE      = 0.10
MUTATION_RATE       = 0.10
MIN_LIVING          = 3


@dataclass
class EvolutionEvent:
    tick        : int
    dead_id     : int
    parent_id   : int
    dead_tokens : int
    parent_tokens: int


@dataclass
class EvolutionReport:
    tick    : int
    events  : list[EvolutionEvent] = field(default_factory=list)

    @property
    def n_replaced(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict:
        return {
            "tick"      : self.tick,
            "n_replaced": self.n_replaced,
            "events"    : [
                {
                    "dead"  : e.dead_id,
                    "parent": e.parent_id,
                    "dead_tokens"  : e.dead_tokens,
                    "parent_tokens": e.parent_tokens,
                }
                for e in self.events
            ],
        }


class EvolutionEngine:
    def __init__(self,
                 engine           : SpeciationEngine,
                 registry         : AgentRegistry,
                 evolution_interval: int   = EVOLUTION_INTERVAL,
                 bottom_pct       : float  = BOTTOM_PERCENTILE,
                 top_pct          : float  = TOP_PERCENTILE,
                 mutation_rate    : float  = MUTATION_RATE,
                 min_living       : int    = MIN_LIVING):
        self.engine            = engine
        self.registry          = registry
        self.evolution_interval = evolution_interval
        self.bottom_pct        = bottom_pct
        self.top_pct           = top_pct
        self.mutation_rate     = mutation_rate
        self.min_living        = min_living
        self.history           : list[EvolutionReport] = []
        self.total_replacements = 0

    def maybe_evolve(self, tick: int, balances: dict[int, int],
                     living: list[int]):
        if tick % self.evolution_interval != 0:
            return None
        if len(living) <= self.min_living:
            return None
        return self._evolve(tick, balances, living)

    def _evolve(self, tick: int, balances: dict[int, int],
                living: list[int]) -> EvolutionReport:
        ranked = sorted(living, key=lambda aid: balances.get(aid, 0))

        n_bottom = max(1, int(len(ranked) * self.bottom_pct))
        n_top    = max(1, int(len(ranked) * self.top_pct))

        bottom = ranked[:n_bottom]
        top    = ranked[-n_top:]

        report = EvolutionReport(tick=tick)

        for dead_id in bottom:
            parent_id = random.choice(top)
            if dead_id == parent_id:
                continue

            self.engine.replace_agent(
                dead_id      = dead_id,
                parent_id    = parent_id,
                current_tick = tick,
                mutation_rate= self.mutation_rate,
            )
            if self.registry is not None:
                self.registry.replace_agent(
                    dead_id,
                    parent_id,
                    current_tick=tick,
                )

            dead_tokens   = balances.get(dead_id, 0)
            parent_tokens = balances.get(parent_id, 0)
            balances[dead_id] = parent_tokens // 2

            report.events.append(EvolutionEvent(
                tick          = tick,
                dead_id       = dead_id,
                parent_id     = parent_id,
                dead_tokens   = dead_tokens,
                parent_tokens = parent_tokens,
            ))
            self.total_replacements += 1

        self.history.append(report)
        return report

    def snapshot(self) -> dict:
        return {
            "total_replacements": self.total_replacements,
            "evolution_interval": self.evolution_interval,
            "mutation_rate"     : self.mutation_rate,
            "n_evolution_ticks" : len(self.history),
        }


if __name__ == "__main__":
    import uuid
    from substrate.economy import ComputeEconomy
    from substrate.registry import AgentRegistry

    N = 10
    engine   = SpeciationEngine(n_agents=N, alpha_scale=0.15, top_n=3, seed=42)
    registry = AgentRegistry(n_agents=N)
    evo      = EvolutionEngine(
        engine            = engine,
        registry          = registry,
        evolution_interval= 200,
        mutation_rate     = 0.10,
    )

    living   = list(range(N))
    balances = {i: 100 + i * 15 for i in range(N)}

    print("=== EVOLUTION ENGINE SMOKE TEST ===")
    print(f"  initial balances: {balances}")

    fps_before = {aid: engine.records[aid].fingerprint.clone() for aid in living}

    report = evo.maybe_evolve(tick=200, balances=balances, living=living)
    assert report is not None, "Evolution did not fire at tick 200"
    assert report.n_replaced >= 1, "No agents replaced"

    print(f"\n  evolution at tick 200:")
    for e in report.events:
        print(f"    agent {e.dead_id} (tokens={e.dead_tokens}) "
              f"← parent {e.parent_id} (tokens={e.parent_tokens})")
        print(f"    new balance: {balances[e.dead_id]}")

    for e in report.events:
        fp_before = fps_before[e.dead_id]
        fp_after  = engine.records[e.dead_id].fingerprint
        sim = torch.dot(fp_before, fp_after).item()
        print(f"    fingerprint similarity dead→new: {sim:.3f} (should be <1.0)")
        assert sim < 1.0, "Fingerprint not mutated"

    no_report = evo.maybe_evolve(tick=201, balances=balances, living=living)
    assert no_report is None, "Evolution fired on wrong tick"
    print("\n  tick 201: correctly skipped")

    print(f"\n  snapshot: {evo.snapshot()}")
    print("\n  RESULT: PASS")