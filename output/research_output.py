import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
import re
from dataclasses import dataclass,field
from typing import Optional

REQUIRED_FIELDS = ["claim", "evidence_needed", "experiment", "falsifiable"]
 
ACTION_VERBS = [
    "run", "measure", "collect", "compare", "test", "count",
    "observe", "record", "calculate", "evaluate", "analyze",
    "track", "monitor", "sample", "classify", "score",
]

@dataclass
class ResearchArtifact:
    artifact_id   : str
    agent_id      : int
    hypothesis    : dict
    raw_text      : str
    quality_score : float
    tick          : int
    coalition_id  : Optional[str] = None
 
    def to_dict(self) -> dict:
        return {
            "artifact_id"   : self.artifact_id,
            "agent_id"      : self.agent_id,
            "coalition_id"  : self.coalition_id,
            "quality_score" : round(self.quality_score, 3),
            "tick"          : self.tick,
            "hypothesis"    : self.hypothesis,
        }
 
    @property
    def claim(self) -> str:
        return self.hypothesis.get("claim", "")
 
    @property
    def falsifiable(self) -> bool:
        return bool(self.hypothesis.get("falsifiable", False))

HYPOTHESIS_TEMPLATES = {
    "agent_specialization": {
        "claim": (
            "Agents that receive task-type-consistent routing will "
            "develop stronger skill fingerprints than randomly routed agents "
            "within 200 task exposures."
        ),
        "evidence_needed": [
            "fingerprint pairwise similarity before and after 200 tasks",
            "task routing accuracy over time",
            "token balance distribution by agent type",
        ],
        "experiment": (
            "Run two societies for 500 ticks: one with fingerprint-based "
            "routing, one with random routing. Measure mean pairwise "
            "fingerprint similarity every 100 ticks. Compare cluster counts."
        ),
        "falsifiable": True,
        "falsification_condition": (
            "If random routing produces equal or better fingerprint divergence "
            "than fingerprint-based routing, hypothesis is rejected."
        ),
    },
 
    "coalition_quality": {
        "claim": (
            "Coalitions of 3 specialists produce higher quality artifacts "
            "than any single generalist agent on tasks requiring all three "
            "output types: code, research, and visual."
        ),
        "evidence_needed": [
            "quality scores of coalition vs solo attempts on same tasks",
            "cross-pollination link counts per artifact",
            "token reward distribution after coalition completion",
        ],
        "experiment": (
            "Present 20 identical hard tasks to both coalitions and solo "
            "agents. Measure artifact quality scores. Compute win rate. "
            "Record cross-pollination bonuses in coalition output."
        ),
        "falsifiable": True,
        "falsification_condition": (
            "If coalition win rate < 60% over 20 tasks, hypothesis is rejected."
        ),
    },
 
    "token_economy": {
        "claim": (
            "The token economy creates sufficient selection pressure for "
            "specialization to emerge within 300 ticks without any "
            "hardcoded role assignment."
        ),
        "evidence_needed": [
            "token balance divergence (std dev over time)",
            "skill fingerprint cluster count at ticks 100, 200, 300",
            "death and replacement event frequency",
        ],
        "experiment": (
            "Run society for 300 ticks. Record token balance std dev every "
            "50 ticks. Count distinct fingerprint clusters via k-means. "
            "Log all death events and parent-child fingerprint similarity."
        ),
        "falsifiable": True,
        "falsification_condition": (
            "If fewer than 3 distinct clusters form by tick 300, "
            "or token std dev stays below 20, hypothesis is rejected."
        ),
    },
 
    "knowledge_distillation": {
        "claim": (
            "A shared episodic memory indexed by FAISS improves solution "
            "quality by at least 20% on previously seen problem types "
            "after 1000 tasks compared to agents without shared memory."
        ),
        "evidence_needed": [
            "quality scores on repeated problem types with vs without memory",
            "FAISS retrieval hit rate over time",
            "knowledge graph node count growth curve",
        ],
        "experiment": (
            "Run two societies for 1000 ticks: one with SharedMemory enabled, "
            "one without. Every 100 ticks, test both on 10 fixed benchmark "
            "problems. Compare mean quality scores. Measure retrieval latency."
        ),
        "falsifiable": True,
        "falsification_condition": (
            "If memory-enabled society scores < 20% higher on benchmark "
            "problems at tick 1000, hypothesis is rejected."
        ),
    },
 
    "web_scraper_topics": {
        "claim": (
            "AI and programming topics collectively dominate Hacker News "
            "front page content, accounting for more than 50% of titles "
            "on any given day."
        ),
        "evidence_needed": [
            "HN front page titles collected across multiple days",
            "topic classification results per title",
            "frequency distribution by topic category",
        ],
        "experiment": (
            "Run scrape_hackernews on HN front page at 3 different times. "
            "Classify each title into: AI, Programming, Business, Science, Other. "
            "Count frequency per category. Compute percentage for AI + Programming."
        ),
        "falsifiable": True,
        "falsification_condition": (
            "If AI + Programming titles account for less than 50% of "
            "scraped titles, hypothesis is rejected."
        ),
    },
 
    "emergent_roles": {
        "claim": (
            "Role specialization emerges from economic selection pressure "
            "alone — no agent is ever explicitly told what role to play."
        ),
        "evidence_needed": [
            "task type distribution per agent over time",
            "fingerprint cluster labels (unsupervised)",
            "auction win rates by task type per agent",
        ],
        "experiment": (
            "Track which task types each agent wins in auctions over 500 ticks. "
            "Apply k-means to fingerprints with k=3. Verify cluster labels "
            "match task-type win distribution without supervision."
        ),
        "falsifiable": True,
        "falsification_condition": (
            "If k-means clusters do not align with task-type win distributions "
            "at accuracy > 70%, hypothesis is rejected."
        ),
    },
}

KEYWORD_MAP = {
    "specializ"     : "agent_specialization",
    "fingerprint"   : "agent_specialization",
    "coalition"     : "coalition_quality",
    "team"          : "coalition_quality",
    "economy"       : "token_economy",
    "token"         : "token_economy",
    "memory"        : "knowledge_distillation",
    "distill"       : "knowledge_distillation",
    "knowledge"     : "knowledge_distillation",
    "scrape"        : "web_scraper_topics",
    "hacker news"   : "web_scraper_topics",
    "hackernews"    : "web_scraper_topics",
    "topic"         : "web_scraper_topics",
    "role"          : "emergent_roles",
    "emergent"      : "emergent_roles",
}

class CoherenceScorer:
    def score(self, hypothesis: dict) -> float:
        try:
            from output.llm_backend import score_coherence, available
            if available():
                return score_coherence(hypothesis)
        except Exception:
            pass
        score = 0.0
        if all(f in hypothesis for f in REQUIRED_FIELDS):
            score += 0.25
        claim = hypothesis.get("claim", "")
        if len(claim.split()) >= 8:
            score += 0.25
        experiment = hypothesis.get("experiment", "").lower()
        if any(v in experiment for v in ACTION_VERBS):
            score += 0.25
        if hypothesis.get("falsifiable") is True:
            score += 0.25
        return round(score, 3)

    def detailed_score(self, hypothesis: dict) -> dict:
        fields_ok  = all(f in hypothesis for f in REQUIRED_FIELDS)
        claim_ok   = len(hypothesis.get("claim", "").split()) >= 8
        exp_ok     = any(
            v in hypothesis.get("experiment", "").lower()
            for v in ACTION_VERBS
        )
        fals_ok    = hypothesis.get("falsifiable") is True
 
        return {
            "total"              : self.score(hypothesis),
            "fields_present"     : fields_ok,
            "claim_nontrivial"   : claim_ok,
            "experiment_concrete": exp_ok,
            "falsifiable"        : fals_ok,
        }

class ResearchOutputLayer:
    def __init__(self):
        self.scorer  = CoherenceScorer()
        self.history : list[ResearchArtifact] = []
        self.total_produced = 0
    def select_template(self, task_desc: str) -> dict:
        desc_lower = task_desc.lower()
        for keyword, template_name in KEYWORD_MAP.items():
            if keyword in desc_lower:
                return HYPOTHESIS_TEMPLATES[template_name]
        
        return HYPOTHESIS_TEMPLATES["agent_specialization"]
    
    def produce(self, agent_id      : int,
                task_desc     : str,
                tick          : int,
                coalition_id  : Optional[str] = None,
                custom_hyp    : Optional[dict] = None,
                difficulty    : float = 0.5) -> ResearchArtifact:
        if custom_hyp:
            hypothesis = custom_hyp
        else:
            try:
                from output.llm_backend import generate_hypothesis, available
                import json
                if available():
                    raw = generate_hypothesis(task_desc, max_tokens=300)
                    parsed = json.loads(raw) if raw else {}
                    if parsed.get("claim") and parsed.get("experiment"):
                        hypothesis = parsed
                    else:
                        hypothesis = self.select_template(task_desc)
                else:
                    hypothesis = self.select_template(task_desc)
            except Exception:
                hypothesis = self.select_template(task_desc)
        raw_text   = json.dumps(hypothesis, indent=2)
        base_score = self.scorer.score(hypothesis)

        import random
        if difficulty < 0.4:
            quality = base_score * random.uniform(0.5, 0.75)
        elif difficulty < 0.75:
            quality = base_score * random.uniform(0.75, 0.95)
        else:
            quality = base_score
        quality = round(min(1.0, quality), 3)
 
        artifact = ResearchArtifact(
            artifact_id   = str(uuid.uuid4()),
            agent_id      = agent_id,
            hypothesis    = hypothesis,
            raw_text      = raw_text,
            quality_score = quality,
            tick          = tick,
            coalition_id  = coalition_id,
        )
 
        self.history.append(artifact)
        self.total_produced += 1
        return artifact
    
    def snapshot(self) -> dict:
        if not self.history:
            return {"total_produced": 0, "mean_quality": 0.0}
        mean_q = sum(a.quality_score for a in self.history) / len(self.history)
        return {
            "total_produced" : self.total_produced,
            "mean_quality"   : round(mean_q, 3),
        }

if __name__ == "__main__":
    print("=" * 56)
    print(" ResearchOutputLayer smoke test")
    print("=" * 56)
 
    layer  = ResearchOutputLayer()
    scorer = CoherenceScorer()
    print("\n── Test 1: web scraper hypothesis ──")
    art = layer.produce(
        agent_id  = 4,
        task_desc = "write a hypothesis about what topics dominate Hacker News",
        tick      = 10,
    )
    print(f"  claim    : {art.claim[:70]}...")
    print(f"  quality  : {art.quality_score}")
    detail = scorer.detailed_score(art.hypothesis)
    for k, v in detail.items():
        print(f"  {k:25s}: {v}")
    assert art.quality_score == 1.0
    assert art.falsifiable
    print("web scraper hypothesis scores 1.0")

    print("\n── Test 2: coalition quality hypothesis ──")
    art2 = layer.produce(
        agent_id  = 5,
        task_desc = "write a hypothesis about coalition formation quality",
        tick      = 11,
    )
    print(f"  claim    : {art2.claim[:70]}...")
    print(f"  quality  : {art2.quality_score}")
    assert art2.quality_score == 1.0
    print("coalition hypothesis scores 1.0")

    print("\n── Test 3: broken hypothesis ──")
    bad_hyp = {"claim": "things happen"}
    art3 = layer.produce(
        agent_id   = 3,
        task_desc  = "some task",
        tick       = 12,
        custom_hyp = bad_hyp,
    )
    print(f"  quality  : {art3.quality_score}")
    detail3 = scorer.detailed_score(bad_hyp)
    print(f"  breakdown: {detail3}")
    assert art3.quality_score < 0.5
    print("broken hypothesis scores below 0.5")

    print("\n── Test 4: all templates ──")
    for name, hyp in HYPOTHESIS_TEMPLATES.items():
        s = scorer.score(hyp)
        print(f"  {'✓' if s==1.0 else '✗'} {name:30s} score={s}")
        assert s == 1.0, f"Template {name} scored {s}"
 
    print("\n── Test 5: cross-pollination content ──")
    scraper_art = layer.produce(
        agent_id  = 4,
        task_desc = "hypothesis about hackernews scraper results",
        tick      = 15,
    )
    raw = scraper_art.raw_text.lower()
    has_ref = any(w in raw for w in ["scrape", "collect", "run", "measure"])
    assert has_ref, "Research output has no cross-pollination terms"
    print(f"  cross-ref terms present: {has_ref}")
    print("✓ cross-pollination content present")
 
    print(f"\nSnapshot: {layer.snapshot()}")
    print("\n" + "=" * 56)
    print("ResearchOutputLayer — DONE")
    print("Next: output/visual_output.py")
    print("=" * 56)






    


    





    


