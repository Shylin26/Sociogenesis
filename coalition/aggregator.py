import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AggregatedOutput:
    output_id       : str
    coalition_id    : str
    content         : str
    quality_score   : float
    cross_refs      : list  = field(default_factory=list)
    cross_ref_bonus : float = 0.0
    member_scores   : dict  = field(default_factory=dict)
    types_present   : set   = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "output_id"       : self.output_id,
            "coalition_id"    : self.coalition_id,
            "quality_score"   : round(self.quality_score, 3),
            "cross_ref_bonus" : round(self.cross_ref_bonus, 3),
            "cross_refs"      : self.cross_refs,
            "types_present"   : list(self.types_present),
            "content_length"  : len(self.content),
        }


class CoalitionAggregator:
    CROSS_REF_BONUS = 0.05
    DIVERSITY_BONUS = 0.05
    MAX_BONUS       = 0.20

    def aggregate(self, coalition) -> AggregatedOutput:
        members = [m for m in coalition.members if m.completed]
        if not members:
            return AggregatedOutput(
                output_id    = str(uuid.uuid4()),
                coalition_id = coalition.coalition_id,
                content      = "",
                quality_score= 0.0,
            )

        by_type = {}
        for m in members:
            by_type[m.subtask_type] = m

        sections = []
        sections.append("# SOCIOGENESIS Coalition Output")
        sections.append(f"# coalition_id: {coalition.coalition_id[:12]}")
        sections.append(f"# coordinator: agent {coalition.coordinator_id}")
        sections.append(f"# members: {[m.agent_id for m in members]}")
        sections.append("")

        if "code" in by_type:
            m = by_type["code"]
            sections.append("## CODE ARTIFACT")
            sections.append(f"# produced by agent {m.agent_id} "
                            f"(quality={m.quality_score:.2f})")
            sections.append(m.output)
            sections.append("")

        if "research" in by_type:
            m = by_type["research"]
            sections.append("## RESEARCH ARTIFACT")
            sections.append(f"# produced by agent {m.agent_id} "
                            f"(quality={m.quality_score:.2f})")
            sections.append(m.output)
            sections.append("")

        if "visual" in by_type:
            m = by_type["visual"]
            sections.append("## VISUAL ARTIFACT")
            sections.append(f"# produced by agent {m.agent_id} "
                            f"(quality={m.quality_score:.2f})")
            sections.append(m.output)
            sections.append("")

        content      = "\n".join(sections)
        cross_refs   = self._detect_cross_refs(by_type)
        mean_quality = sum(m.quality_score for m in members) / len(members)
        bonus        = self._compute_bonus(by_type, cross_refs)
        final_quality= min(1.0, mean_quality + bonus)
        member_scores= {m.agent_id: m.quality_score for m in members}
        types_present= set(by_type.keys())

        return AggregatedOutput(
            output_id       = str(uuid.uuid4()),
            coalition_id    = coalition.coalition_id,
            content         = content,
            quality_score   = round(final_quality, 3),
            cross_refs      = cross_refs,
            cross_ref_bonus = round(bonus, 3),
            member_scores   = member_scores,
            types_present   = types_present,
        )

    def _detect_cross_refs(self, by_type: dict) -> list:
        refs = []

        code_output     = by_type.get("code",     None)
        research_output = by_type.get("research", None)
        visual_output   = by_type.get("visual",   None)

        code_terms = self._extract_code_terms(code_output.output) \
                     if code_output else set()

        if research_output and code_terms:
            research_text = research_output.output.lower()
            for term in code_terms:
                if term.lower() in research_text:
                    refs.append(("research", "code", term))
                    break

        if visual_output and code_terms:
            visual_text = visual_output.output.lower()
            for term in code_terms:
                if term.lower() in visual_text:
                    refs.append(("visual", "code", term))
                    break

        if visual_output and research_output:
            research_terms = self._extract_research_terms(
                research_output.output
            )
            visual_text = visual_output.output.lower()
            for term in research_terms:
                if term.lower() in visual_text:
                    refs.append(("visual", "research", term))
                    break

        return refs

    def _extract_code_terms(self, code_output: str) -> set:
        terms = set()
        for match in re.finditer(r'def\s+(\w+)', code_output):
            terms.add(match.group(1))
        for match in re.finditer(r'class\s+(\w+)', code_output):
            terms.add(match.group(1))
        for match in re.finditer(r'^(\w+)\s*=', code_output, re.MULTILINE):
            name = match.group(1)
            if len(name) > 3:
                terms.add(name)
        return terms

    def _extract_research_terms(self, research_output: str) -> set:
        from collections import Counter
        words = re.findall(r'\b\w{7,}\b', research_output)
        freq  = Counter(w.lower() for w in words)
        return {w for w, _ in freq.most_common(10)}

    def _compute_bonus(self, by_type: dict, cross_refs: list) -> float:
        bonus = len(cross_refs) * self.CROSS_REF_BONUS
        if len(by_type) == 3:
            bonus += self.DIVERSITY_BONUS
        return min(self.MAX_BONUS, bonus)

    def compare_coalition_vs_solo(self,
                                  coalition_quality : float,
                                  solo_qualities    : list[float]) -> dict:
        best_solo = max(solo_qualities) if solo_qualities else 0.0
        mean_solo = sum(solo_qualities) / len(solo_qualities) \
                    if solo_qualities else 0.0
        coalition_wins = coalition_quality > best_solo
        margin         = coalition_quality - best_solo

        return {
            "coalition_quality" : round(coalition_quality, 3),
            "best_solo_quality" : round(best_solo, 3),
            "mean_solo_quality" : round(mean_solo, 3),
            "coalition_wins"    : coalition_wins,
            "margin"            : round(margin, 3),
        }


if __name__ == "__main__":
    from coalition.task_decomposer import TaskDecomposer
    from coalition.Auction         import AuctionEngine
    from coalition.coalition       import CoalitionFormation
    import torch
    import torch.nn.functional as F

    print("=" * 56)
    print("PANTHEON Week 4 — Aggregator smoke test")
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

    decomp  = TaskDecomposer(min_types=3).decompose_demo_task()
    results = AuctionEngine().run_all_auctions(
        subtasks       = decomp.subtasks,
        living_agents  = list(range(N_AGENTS)),
        fingerprints   = fingerprints,
        token_balances = balances,
        type_seeds     = type_seeds,
    )

    cf = CoalitionFormation()
    coalition = cf.form(
        decomp.task_id, results, decomp.subtasks,
        reputations, private_latents, balances, tick=10
    )
    assert coalition is not None

    code_output = (
        "def scrape_hackernews(url):\n"
        "    import requests\n"
        "    from bs4 import BeautifulSoup\n"
        "    response = requests.get(url)\n"
        "    soup = BeautifulSoup(response.text, 'html.parser')\n"
        "    titles = [t.text for t in soup.find_all('a', class_='titlelink')]\n"
        "    return titles\n"
        "\n"
        "results = scrape_hackernews('https://news.ycombinator.com')\n"
    )
    research_output = (
        "Hypothesis: AI and programming topics will dominate HN today.\n"
        "Evidence needed: run scrape_hackernews and count topic categories.\n"
        "Experiment: classify each title into tech/AI/business/other.\n"
        "Falsifiable: if AI topics < 30% of results, hypothesis rejected.\n"
    )
    visual_output = (
        "Data flow diagram of scrape_hackernews architecture:\n"
        "  [HTTP Request] → [BeautifulSoup Parser] → [Title Extractor]\n"
        "  → [results list] → [Topic Classifier] → [Bar Chart]\n"
        "The scrape_hackernews function is the entry point.\n"
    )

    for member in coalition.members:
        if member.subtask_type == "code":
            cf.record_output(coalition.coalition_id, member.agent_id,
                             code_output, 0.88, 20)
        elif member.subtask_type == "research":
            cf.record_output(coalition.coalition_id, member.agent_id,
                             research_output, 0.75, 20)
        elif member.subtask_type == "visual":
            cf.record_output(coalition.coalition_id, member.agent_id,
                             visual_output, 0.70, 20)

    print("\n── Test 1: aggregate outputs ──")
    agg    = CoalitionAggregator()
    output = agg.aggregate(coalition)

    print(f"  quality_score   : {output.quality_score}")
    print(f"  cross_ref_bonus : {output.cross_ref_bonus}")
    print(f"  cross_refs      : {output.cross_refs}")
    print(f"  types_present   : {output.types_present}")
    print(f"  content length  : {len(output.content)} chars")

    assert output.quality_score > 0
    assert len(output.types_present) == 3
    print("aggregation complete with all 3 types")

    print("\n── Test 2: cross-pollination ──")
    assert len(output.cross_refs) > 0, "No cross-refs detected"
    print(f"  detected refs: {output.cross_refs}")
    assert output.cross_ref_bonus > 0
    print(f"cross-pollination bonus applied: +{output.cross_ref_bonus}")

    print("\n── Test 3: coalition vs solo ──")
    solo_qualities = [0.55, 0.48, 0.62, 0.51, 0.44]
    comparison = agg.compare_coalition_vs_solo(
        output.quality_score, solo_qualities
    )
    print(f"  coalition: {comparison['coalition_quality']}")
    print(f"  best solo: {comparison['best_solo_quality']}")
    print(f"  margin   : {comparison['margin']}")
    print(f"  coalition wins: {comparison['coalition_wins']}")
    assert comparison["coalition_wins"], "Coalition should beat solo"
    print("coalition beats best solo agent")

    completed = cf.complete(
        coalition.coalition_id, output.content, tick=25
    )
    print(f"\nReward distribution:")
    for m in completed.members:
        print(f"  agent {m.agent_id} [{m.subtask_type:8s}]: "
              f"quality={m.quality_score:.2f} "
              f"reward={m.reward_share} tokens")

    print("\n" + "=" * 56)
    print("Aggregator — DONE")
    print("Next: coalition/week4_loop.py")
    print("=" * 56)
