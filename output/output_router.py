import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F
from output.code_output     import CodeOutputLayer,     CodeArtifact
from output.research_output import ResearchOutputLayer, ResearchArtifact
from output.visual_output   import VisualOutputLayer,   VisualArtifact
from memory.episodic        import EpisodicMemory
import numpy as np

RAG_BONUS_CAP   = 0.15
RAG_BONUS_SCALE = 0.20
@dataclass
class RoutedOutput:
    coalition_id    : str
    task_desc       : str
    tick            : int
    code_artifact   : Optional[CodeArtifact]     = None
    research_artifact: Optional[ResearchArtifact]= None
    visual_artifact : Optional[VisualArtifact]   = None

    @property
    def mean_quality(self) -> float:
        scores = []
        if self.code_artifact:
            scores.append(self.code_artifact.quality_score)
        if self.research_artifact:
            scores.append(self.research_artifact.quality_score)
        if self.visual_artifact:
            scores.append(self.visual_artifact.quality_score)
        return round(sum(scores) / len(scores), 3) if scores else 0.0
 
    @property
    def all_present(self) -> bool:
        return all([self.code_artifact,
                    self.research_artifact,
                    self.visual_artifact])
 
    def to_dict(self) -> dict:
        return {
            "coalition_id" : self.coalition_id,
            "task_desc"    : self.task_desc[:80],
            "tick"         : self.tick,
            "mean_quality" : self.mean_quality,
            "all_present"  : self.all_present,
            "code_quality" : self.code_artifact.quality_score
                             if self.code_artifact else None,
            "research_quality": self.research_artifact.quality_score
                             if self.research_artifact else None,
            "visual_quality": self.visual_artifact.quality_score
                             if self.visual_artifact else None,
        }

class OutputRouter:
    EXECUTION_ORDER = ["code", "research", "visual"]
 
    def __init__(self, visual_mode: str = "ascii",
                 episodic: Optional[EpisodicMemory] = None):
        self.code_layer     = CodeOutputLayer(timeout=5)
        self.research_layer = ResearchOutputLayer()
        self.visual_layer   = VisualOutputLayer(mode=visual_mode)
        self.history        : list[RoutedOutput] = []
        self.episodic       = episodic
 
    def route_and_produce(self,
                          coalition,
                          task_desc    : str,
                          tick         : int,
                          fingerprints : dict[int, torch.Tensor],
                          type_seeds   : dict[str, torch.Tensor],
                          difficulty   : float = 0.5,
                          task_emb     : Optional[np.ndarray] = None) -> RoutedOutput:

        context = {
            "title"            : self._extract_title(task_desc),
            "code_fn_name"     : "",
            "code_var_name"    : "",
            "research_claim"   : "",
        }
        assignments = self._assign_types(coalition, fingerprints, type_seeds)
        output = RoutedOutput(
            coalition_id = coalition.coalition_id,
            task_desc    = task_desc,
            tick         = tick,
        )

        for output_type in self.EXECUTION_ORDER:
            agent_id = assignments.get(output_type)
            if agent_id is None:
                continue
 
            if output_type == "code":
                art = self._produce_code(
                    agent_id, task_desc, tick,
                    coalition.coalition_id, context, difficulty
                )
                output.code_artifact = art
            elif output_type == "research":
                art = self._produce_research(
                    agent_id, task_desc, tick,
                    coalition.coalition_id, context, difficulty
                )
                output.research_artifact = art
            elif output_type == "visual":
                art = self._produce_visual(
                    agent_id, task_desc, tick,
                    coalition.coalition_id, context, difficulty
                )
                output.visual_artifact = art

            if art is not None and task_emb is not None and self.episodic is not None:
                art.quality_score = self._apply_rag_bonus(art.quality_score, task_emb)
               

 
        self.history.append(output)
        return output
    
    def _apply_rag_bonus(self, base_quality: float,
                         task_emb: np.ndarray) -> float:
        hits = self.episodic.retrieve(task_emb, k=3)
        if not hits:
            return base_quality
        mean_hit_quality = float(np.mean([
            getattr(h, "quality", getattr(h, "quality_score", 0.0))
            for h in hits
        ]))
        bonus = min(RAG_BONUS_CAP, mean_hit_quality * RAG_BONUS_SCALE)
        return min(1.0, base_quality + bonus)

    def _assign_types(self, coalition,
                      fingerprints : dict[int, torch.Tensor],
                      type_seeds   : dict[str, torch.Tensor]) -> dict[str, int]:

        assigned    = {}  
        used_agents = set()
 
        for output_type in self.EXECUTION_ORDER:
            seed = type_seeds.get(output_type)
            if seed is None:
                continue
 
            best_id    = None
            best_score = -999.0
 
            for member in coalition.members:
                if member.agent_id in used_agents:
                    continue
                fp = fingerprints.get(member.agent_id)
                if fp is None:
                    continue
                score = torch.dot(
                    F.normalize(seed.float(), dim=0),
                    F.normalize(fp.float(), dim=0)
                ).item()
                if score > best_score:
                    best_score = score
                    best_id    = member.agent_id
 
            if best_id is not None:
                assigned[output_type] = best_id
                used_agents.add(best_id)
 
        return assigned 
    
    def _produce_code(self, agent_id: int, task_desc: str,
                      tick: int, coalition_id: str,
                      context: dict, difficulty: float = 0.5) -> CodeArtifact:
        art = self.code_layer.produce(
            agent_id     = agent_id,
            task_desc    = task_desc,
            tick         = tick,
            coalition_id = coalition_id,
            difficulty   = difficulty,
        )
 
        import re
        fns = re.findall(r'def\s+(\w+)', art.code)
        if fns:
            context["code_fn_name"]  = fns[0]
            context["code_var_name"] = fns[0].replace("def ", "") + "_results"
 
        return art
    
    def _produce_research(self, agent_id: int, task_desc: str,
                           tick: int, coalition_id: str,
                           context: dict, difficulty: float = 0.5) -> ResearchArtifact:
        enriched_desc = task_desc
        if context.get("code_fn_name"):
            enriched_desc = (
                f"{task_desc} "
                f"(code function: {context['code_fn_name']})"
            )

        art = self.research_layer.produce(
            agent_id     = agent_id,
            task_desc    = enriched_desc,
            tick         = tick,
            coalition_id = coalition_id,
            difficulty   = difficulty,
        )

        context["research_claim"] = art.claim[:60]
        return art
    
    def _produce_visual(self, agent_id: int, task_desc: str,
                         tick: int, coalition_id: str,
                         context: dict, difficulty: float = 0.5) -> VisualArtifact:
        from output.visual_output import extract_components
        task_components, task_title = extract_components(task_desc)
        visual_context = {
            "title": context.get("title", task_title),
            "components": task_components,
            "task_desc": task_desc,
        }
        if context.get("code_fn_name"):
            fn = context["code_fn_name"]
            visual_context["components"] = [
                "HTTP Request",
                f"{fn}(url)",
                "HTML Parser",
                f"{fn}_results [ ]",
                "Topic Classifier",
                "Output Chart",
            ]

        art = self.visual_layer.produce(
            agent_id     = agent_id,
            task_desc    = task_desc,
            tick         = tick,
            coalition_id = coalition_id,
            context      = visual_context,
            difficulty   = difficulty,
        )
        return art

    def _extract_title(self, task_desc: str) -> str:

        words = task_desc.split()
        return " ".join(words[:5]) if len(words) >= 5 else task_desc
 
    
    def snapshot(self) -> dict:
        if not self.history:
            return {"total_routed": 0, "mean_quality": 0.0}
        mean_q = sum(o.mean_quality for o in self.history) / len(self.history)
        all_3  = sum(1 for o in self.history if o.all_present)
        return {
            "total_routed" : len(self.history),
            "mean_quality" : round(mean_q, 3),
            "all_3_types"  : all_3,
            "code"         : self.code_layer.snapshot(),
            "research"     : self.research_layer.snapshot(),
            "visual"       : self.visual_layer.snapshot(),
        }    
                        
if __name__ == "__main__":
    from coalition.task_decomposer import TaskDecomposer
    from coalition.Auction         import AuctionEngine
    from coalition.coalition       import CoalitionFormation
 
    print("=" * 56)
    print("OutputRouter smoke test")
    print("=" * 56)
 
    N_AGENTS   = 10
    TASK_TYPES = ["code", "research", "visual"]

    gen = torch.Generator(); gen.manual_seed(42)
    type_seeds = {
        name: F.normalize(torch.randn(128, generator=gen), dim=0)
        for name in TASK_TYPES
    }
 
    gen2 = torch.Generator(); gen2.manual_seed(7)
    fingerprints = {}
    for i in range(N_AGENTS):
        base = (type_seeds["code"]     if i < 3 else
                type_seeds["research"] if i < 6 else
                type_seeds["visual"])
        fingerprints[i] = F.normalize(
            base + torch.randn(128, generator=gen2) * 0.1, dim=0
        )
 
    balances    = {i: 100 for i in range(N_AGENTS)}
    reputations = {i: float(i) * 0.1 for i in range(N_AGENTS)}
    gen3 = torch.Generator(); gen3.manual_seed(99)
    private_latents = {
        i: F.normalize(torch.randn(128, generator=gen3), dim=0)
        for i in range(N_AGENTS)
    }

    task_desc = (
        "Build a Python web scraper for Hacker News front page. "
        "Write a hypothesis about what topics dominate today. "
        "Generate a data flow diagram of the scraper architecture."
    )
    decomp    = TaskDecomposer(min_types=3).decompose(
        task_desc, str(uuid.uuid4()), difficulty=0.85
    )
    results   = AuctionEngine().run_all_auctions(
        subtasks=decomp.subtasks, living_agents=list(range(N_AGENTS)),
        fingerprints=fingerprints, token_balances=balances,
        type_seeds=type_seeds,
    )
    cf        = CoalitionFormation()
    coalition = cf.form(
        decomp.task_id, results, decomp.subtasks,
        reputations, private_latents, balances, tick=10
    )
    assert coalition is not None, "Coalition formation failed"
    print(f"\nCoalition formed: members={coalition.member_ids} "
          f"coordinator={coalition.coordinator_id}")

    print("\n── Test 1: route and produce all outputs ──")
    router = OutputRouter(visual_mode="ascii")
    output = router.route_and_produce(
        coalition    = coalition,
        task_desc    = task_desc,
        tick         = 10,
        fingerprints = fingerprints,
        type_seeds   = type_seeds,
    )
 
    print(f"\n  mean quality  : {output.mean_quality}")
    print(f"  all 3 present : {output.all_present}")
    if output.code_artifact:
        print(f"  code quality  : {output.code_artifact.quality_score}")
        print(f"  code runs     : {output.code_artifact.result.success}")
    if output.research_artifact:
        print(f"  research qual : {output.research_artifact.quality_score}")
        print(f"  claim         : {output.research_artifact.claim[:60]}...")
    if output.visual_artifact:
        print(f"  visual qual   : {output.visual_artifact.quality_score}")
        print(f"  visual mode   : {output.visual_artifact.mode}")
 
    assert output.all_present, "Not all 3 output types produced"
    assert output.code_artifact.result.success, "Code did not run"
    assert output.research_artifact.quality_score == 1.0, "Research not coherent"
    assert output.visual_artifact.quality_score >= 0.3, "Visual quality too low"
    print("all 3 outputs produced and quality checks pass")
 
    print("\n── Test 2: cross-pollination ──")
    code_fns = set()
    import re
    for match in re.finditer(r'def\s+(\w+)', output.code_artifact.code):
        code_fns.add(match.group(1))
 
    visual_content  = output.visual_artifact.content.lower()
    research_content= output.research_artifact.raw_text.lower()
 
    code_in_visual   = any(fn.lower() in visual_content for fn in code_fns)
    code_in_research = any(fn.lower() in research_content for fn in code_fns)
 
    print(f"  code fn names      : {code_fns}")
    print(f"  fn in visual       : {code_in_visual}")
    print(f"  fn in research     : {code_in_research}")
 
    assert code_in_visual or code_in_research, \
        "No cross-pollination detected"
    print("cross-pollination working")

    print("\n── Test 3: specialist assignment ──")
    assignments = router._assign_types(coalition, fingerprints, type_seeds)
    print(f"  assignments: {assignments}")
    for output_type, agent_id in assignments.items():
        expected = {"code": range(3), "research": range(3,6),
                    "visual": range(6,10)}
        if output_type in expected:
            assert agent_id in expected[output_type], \
                f"{output_type} assigned to non-specialist agent {agent_id}"
            print(f"{output_type:8s} → agent {agent_id} (correct specialist)")
 
    print(f"\nSnapshot: {router.snapshot()}")
    print("\n" + "=" * 56)
    print("OutputRouter — DONE")
    print("Next: output/week5_loop.py")
    print("=" * 56)
 


