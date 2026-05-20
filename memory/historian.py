import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass, field
from typing import Optional

from memory.episodic import EpisodicMemory
from memory.distillation import KnowledgeDistiller
from core.society_model import SocietyModel, SocietyEvent
from speciation.engine import SpeciationEngine


REPORT_INTERVAL = 100


@dataclass
class HistorianReport:
    tick              : int
    top_agents        : list[int]
    emerging_roles    : dict[str, list[int]]
    bottlenecks       : list[str]
    recommendations   : list[str]
    episodic_count    : int
    semantic_nodes    : int
    avg_quality       : float
    coalition_count   : int
    solo_count        : int
    evolution_count   : int

    def to_dict(self) -> dict:
        return {
            "tick"           : self.tick,
            "top_agents"     : self.top_agents,
            "emerging_roles" : self.emerging_roles,
            "bottlenecks"    : self.bottlenecks,
            "recommendations": self.recommendations,
            "episodic_count" : self.episodic_count,
            "semantic_nodes" : self.semantic_nodes,
            "avg_quality"    : round(self.avg_quality, 3),
            "coalition_count": self.coalition_count,
            "solo_count"     : self.solo_count,
            "evolution_count": self.evolution_count,
        }

    def summary(self) -> str:
        lines = [
            f"── Historian Report @ tick {self.tick} ──────────────────",
            f"  top agents       : {self.top_agents}",
            f"  emerging roles   : {self.emerging_roles}",
            f"  bottlenecks      : {self.bottlenecks}",
            f"  recommendations  : {self.recommendations}",
            f"  episodic records : {self.episodic_count}",
            f"  semantic nodes   : {self.semantic_nodes}",
            f"  avg quality      : {self.avg_quality:.3f}",
            f"  coalition/solo   : {self.coalition_count}/{self.solo_count}",
            f"  evolution events : {self.evolution_count}",
        ]
        return "\n".join(lines)


class HistorianAgent:
    AGENT_ID = 99

    def __init__(self,
                 episodic         : EpisodicMemory,
                 distiller        : KnowledgeDistiller,
                 engine           : SpeciationEngine,
                 society_model    : SocietyModel,
                 report_interval  : int = REPORT_INTERVAL):
        self.episodic        = episodic
        self.distiller       = distiller
        self.engine          = engine
        self.society_model   = society_model
        self.report_interval = report_interval

        self._reports        : list[HistorianReport] = []
        self._quality_log    : list[float] = []
        self._coalition_count = 0
        self._solo_count      = 0
        self._evolution_count = 0
        self._balances_ref    : dict[int, int] = {}
        self._last_report_tick = -report_interval

    def link_balances(self, balances: dict[int, int]):
        self._balances_ref = balances

    def log_task(self, quality: float, is_coalition: bool):
        self._quality_log.append(quality)
        if is_coalition:
            self._coalition_count += 1
        else:
            self._solo_count += 1

    def log_evolution(self):
        self._evolution_count += 1

    def on_tick(self, tick: int) -> Optional[HistorianReport]:
        if tick - self._last_report_tick < self.report_interval:
            return None
        self._last_report_tick = tick
        report = self._generate_report(tick)
        self._reports.append(report)
        return report

    def _generate_report(self, tick: int) -> HistorianReport:
        top_agents     = self._top_agents()
        emerging_roles = self._emerging_roles()
        bottlenecks    = self._detect_bottlenecks()
        recommendations= self._recommendations(bottlenecks)
        avg_quality    = (
            sum(self._quality_log) / len(self._quality_log)
            if self._quality_log else 0.0
        )

        return HistorianReport(
            tick              = tick,
            top_agents        = top_agents,
            emerging_roles    = emerging_roles,
            bottlenecks       = bottlenecks,
            recommendations   = recommendations,
            episodic_count    = len(self.episodic),
            semantic_nodes    = self.distiller.node_count,
            avg_quality       = avg_quality,
            coalition_count   = self._coalition_count,
            solo_count        = self._solo_count,
            evolution_count   = self._evolution_count,
        )

    def _top_agents(self, n: int = 3) -> list[int]:
        if not self._balances_ref:
            return []
        ranked = sorted(
            self._balances_ref.items(), key=lambda x: x[1], reverse=True
        )
        return [aid for aid, _ in ranked[:n]]

    def _emerging_roles(self) -> dict[str, list[int]]:
        roles: dict[str, list[int]] = {}
        for aid, rec in self.engine.records.items():
            fp    = rec.fingerprint
            best  = max(
                self.engine._type_seeds.items(),
                key=lambda kv: float(fp @ kv[1])
            )
            role  = best[0]
            roles.setdefault(role, []).append(aid)
        return roles

    def _detect_bottlenecks(self) -> list[str]:
        bottlenecks = []
        roles = self._emerging_roles()
        for role in ["code", "research", "visual"]:
            agents = roles.get(role, [])
            if len(agents) == 0:
                bottlenecks.append(f"no {role} specialists")
            elif len(agents) == 1:
                bottlenecks.append(f"single {role} specialist (agent {agents[0]})")
        if self._quality_log:
            recent = self._quality_log[-50:]
            if sum(recent) / len(recent) < 0.5:
                bottlenecks.append("quality dropping below 0.5 threshold")
        snap = self.society_model.snapshot()
        if snap["train_steps"] > 0 and snap["avg_loss"] > 3.0:
            bottlenecks.append("society model loss high — prediction unreliable")
        return bottlenecks

    def _recommendations(self, bottlenecks: list[str]) -> list[str]:
        recs = []
        for b in bottlenecks:
            if "no code" in b:
                recs.append("seed a code-specialist agent")
            elif "no research" in b:
                recs.append("seed a research-specialist agent")
            elif "no visual" in b:
                recs.append("seed a visual-specialist agent")
            elif "single" in b:
                recs.append(f"diversify: {b.replace('single ', '')}")
            elif "quality" in b:
                recs.append("increase task difficulty variance")
            elif "society model" in b:
                recs.append("feed more events to society model")
        if not recs:
            recs.append("society healthy — no action needed")
        return recs

    def all_reports(self) -> list[dict]:
        return [r.to_dict() for r in self._reports]

    def latest_report(self) -> Optional[HistorianReport]:
        return self._reports[-1] if self._reports else None

    def snapshot(self) -> dict:
        return {
            "total_reports"   : len(self._reports),
            "coalition_count" : self._coalition_count,
            "solo_count"      : self._solo_count,
            "evolution_count" : self._evolution_count,
            "avg_quality"     : round(
                sum(self._quality_log) / len(self._quality_log), 3
            ) if self._quality_log else 0.0,
        }


if __name__ == "__main__":
    import random
    import numpy as np
    from memory.distillation import KnowledgeDistiller
    from memory.librarian import LibrarianAgent

    print("=== HISTORIAN AGENT SMOKE TEST ===")

    N = 10
    engine       = SpeciationEngine(n_agents=N, alpha_scale=0.15, top_n=3, seed=42)
    episodic     = EpisodicMemory()
    distiller    = KnowledgeDistiller(episodic=episodic)
    society_model= SocietyModel(d_model=64, n_heads=4, n_layers=2, ctx=32)
    historian    = HistorianAgent(
        episodic      = episodic,
        distiller     = distiller,
        engine        = engine,
        society_model = society_model,
        report_interval = 100,
    )

    balances = {i: 100 + i * 10 for i in range(N)}
    historian.link_balances(balances)

    print("  simulating 200 ticks of logs...")
    for tick in range(1, 201):
        quality      = random.uniform(0.2, 1.0)
        is_coalition = tick % 3 == 0
        historian.log_task(quality, is_coalition)

        if tick % 200 == 0:
            historian.log_evolution()

        ev = SocietyEvent(
            tick       = tick,
            event_type = random.choice(["TASK_SUCCESS", "TASK_FAIL",
                                        "COALITION_FORMED", "AGENT_DIED"]),
            agent_id   = random.randint(0, N - 1),
            quality    = quality,
            success    = quality > 0.5,
        )
        society_model.observe(ev)

        report = historian.on_tick(tick)
        if report:
            print(f"\n{report.summary()}")

    assert len(historian._reports) >= 2, "Expected at least 2 reports"
    latest = historian.latest_report()
    assert latest is not None
    assert latest.episodic_count >= 0
    assert len(latest.emerging_roles) > 0
    print(f"\n  snapshot: {historian.snapshot()}")
    print("\n  RESULT: PASS")