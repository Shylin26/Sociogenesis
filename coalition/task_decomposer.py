import re
import uuid
from dataclasses import dataclass,field
from enum import Enum
from typing import Optional

class SubtaskType(Enum):
    CODE="code"
    RESEARCH="research"
    VISUAL="visual"

@dataclass
class Subtask:
    subtask_id:str
    parent_id:str
    task_type:SubtaskType
    description:str
    difficulty: float
    reward: int =0
    depends_on:list=field(default_factory=list)

    def __post_init__(self):
        if self.reward==0:
            self.reward=max(1,int(self.difficulty*20))
        self.difficulty=max(0.0,min(1.0,self.difficulty))
    
    def to_dict(self)->dict:
        return{
            "subtask_id":self.subtask_id,
            "parent_id":self.parent_id,
            "task_type":self.task_type.value,
            "description":self.description,
            "difficulty":round(self.difficulty,2),
            "reward": self.reward,
            "depends_on":self.depends_on,
        }

@dataclass
class DecompositionResult:
    subtasks:list[Subtask]
    confidence:float
    method:str
    rationale:str
    task_id:str

    def __repr__(self):
        lines = [f"DecompositionResult(confidence={self.confidence:.2f}"]
        for s in self.subtasks:
            lines.append(f"  [{s.task_type.value:8s}] {s.description[:60]}")
        return "\n".join(lines) + "\n)"

CODE_PATTERNS = [
    (["scraper", "scrape", "crawl", "fetch", "requests", "beautifulsoup"],
     "Write a Python web scraper for the described target. "
     "Return working code that fetches and parses the data."),
 
    (["sort", "search", "algorithm", "implement", "function", "code",
      "write", "build", "program", "script"],
     "Implement the described algorithm or function in Python. "
     "Include a test case that verifies correctness."),
 
    (["api", "endpoint", "rest", "http", "server", "flask", "fastapi"],
     "Build the described API endpoint or server component in Python. "
     "Include route definition and basic error handling."),
 
    (["data", "parse", "process", "clean", "transform", "pipeline"],
     "Write a Python data processing pipeline for the described task. "
     "Handle edge cases and return structured output."),
 
    (["model", "train", "neural", "ml", "machine learning", "classifier"],
     "Implement the described ML model or training loop in Python. "
     "Use standard libraries (torch, sklearn). Include evaluation code."),
]
RESEARCH_PATTERNS = [
    (["hypothesis", "claim", "theory", "predict", "expect"],
     "Write a structured research hypothesis about the described topic. "
     "Include: claim, evidence_needed, experiment_design, falsifiability."),
 
    (["analyze", "analyse", "study", "investigate", "explore", "understand"],
     "Write a research analysis of the described topic. "
     "Include key findings, open questions, and a testable prediction."),
 
    (["compare", "versus", "vs", "difference", "contrast"],
     "Write a structured comparison of the described items. "
     "Include: key dimensions, trade-offs, and a recommendation."),
 
    (["why", "what", "how", "explain", "describe", "summarize", "survey"],
     "Write a concise research summary of the described topic. "
     "Include: core mechanism, current understanding, open questions."),
 
    (["topic", "dominate", "trend", "pattern", "behavior", "find"],
     "Formulate a falsifiable hypothesis about what patterns will be found. "
     "Include: prediction, methodology, success criteria."),
]
VISUAL_PATTERNS = [
    (["diagram", "flow", "architecture", "structure", "layout"],
     "Create a data flow diagram of the described system or architecture. "
     "Show components, connections, and data direction."),
 
    (["chart", "plot", "graph", "visualize", "visualise", "show"],
     "Generate a chart or plot visualizing the described data or results. "
     "Choose the most appropriate chart type for the data."),
 
    (["t-sne", "tsne", "embedding", "cluster", "scatter"],
     "Generate a t-SNE or scatter plot visualizing the described embeddings. "
     "Color-code by category. Include axis labels and legend."),
 
    (["network", "graph", "node", "edge", "relationship"],
     "Create a network graph visualizing the described relationships. "
     "Use force-directed layout. Node size = importance."),
 
    (["timeline", "history", "sequence", "over time", "progress"],
     "Create a timeline visualization of the described sequence of events. "
     "Show key milestones and durations."),
]
FALLBACK_TEMPLATES = {
    SubtaskType.CODE: (
        "Implement a Python solution for the core computational "
        "component of this task. Return working, tested code."
    ),
    SubtaskType.RESEARCH: (
        "Write a structured research hypothesis about this task. "
        "Include claim, evidence needed, and falsifiability condition."
    ),
    SubtaskType.VISUAL: (
        "Create a visual representation (diagram, chart, or graph) "
        "that illustrates the key outputs or structure of this task."
    ),
}

class TaskDecomposer:
    def __init__(self, min_subtasks : int = 2,
                 max_subtasks : int = 5,
                 min_types    : int = 2):
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks
        self.min_types    = min_types
    
    def decompose(self, task_description : str,
                  task_id        : str,
                  difficulty     : float = 0.7) -> DecompositionResult:
        """
        Decompose a hard task into subtasks.
 
        Algorithm:
          1. Lowercase and normalize the description
          2. Match against CODE_PATTERNS, RESEARCH_PATTERNS, VISUAL_PATTERNS
          3. For each matched type, instantiate one subtask
          4. If fewer than min_types matched, add fallback subtasks
          5. Compute confidence from match quality
          6. Return DecompositionResult
 
        difficulty is used to set subtask difficulty.
        Hard tasks (0.8+) generate harder subtasks.
        """
        difficulty = max(0.0, min(1.0, difficulty))
        text       = task_description.lower()
        subtasks   = []
        matched_types = set()
        rationale_parts = []

        code_desc, code_conf = self._match_patterns(text, CODE_PATTERNS)
        if code_conf > 0:
            subtasks.append(self._make_subtask(
                parent_id   = task_id,
                task_type   = SubtaskType.CODE,
                description = code_desc,
                difficulty  = min(1.0, difficulty * 0.95),
            ))
            matched_types.add(SubtaskType.CODE)
            rationale_parts.append(
                f"code subtask matched (conf={code_conf:.2f})"
            )

        research_desc, research_conf = self._match_patterns(
            text, RESEARCH_PATTERNS
        )
        if research_conf > 0:
            subtasks.append(self._make_subtask(
                parent_id   = task_id,
                task_type   = SubtaskType.RESEARCH,
                description = research_desc,
                difficulty  = min(1.0, difficulty * 0.85),
            ))
            matched_types.add(SubtaskType.RESEARCH)
            rationale_parts.append(
                f"research subtask matched (conf={research_conf:.2f})"
            )

        visual_desc, visual_conf = self._match_patterns(
            text, VISUAL_PATTERNS
        )
        if visual_conf > 0:
            subtasks.append(self._make_subtask(
                parent_id   = task_id,
                task_type   = SubtaskType.VISUAL,
                description = visual_desc,
                difficulty  = min(1.0, difficulty * 0.80),
            ))
            matched_types.add(SubtaskType.VISUAL)
            rationale_parts.append(
                f"visual subtask matched (conf={visual_conf:.2f})"
            )

        all_types = [SubtaskType.CODE,
                     SubtaskType.RESEARCH,
                     SubtaskType.VISUAL]
        missing   = [t for t in all_types if t not in matched_types]
 
        while len(matched_types) < self.min_types and missing:
            t = missing.pop(0)
            subtasks.append(self._make_subtask(
                parent_id   = task_id,
                task_type   = t,
                description = FALLBACK_TEMPLATES[t],
                difficulty  = difficulty * 0.75,   # fallbacks are easier
            ))
            matched_types.add(t)
            rationale_parts.append(
                f"{t.value} subtask added via fallback (min_types={self.min_types})"
            )

        subtasks = subtasks[:self.max_subtasks]
 

        individual_confs = [code_conf, research_conf, visual_conf]
        matched_confs    = [c for c in individual_confs if c > 0]
        if matched_confs:
            base_conf = sum(matched_confs) / len(matched_confs)
        else:
            base_conf = 0.3   
 
        n_fallback = sum(
            1 for p in rationale_parts if "fallback" in p
        )
        confidence = max(0.2, base_conf - 0.15 * n_fallback)
 
        rationale = "; ".join(rationale_parts) if rationale_parts else \
                    "no keywords matched — all fallbacks used"
 
        return DecompositionResult(
            subtasks   = subtasks,
            confidence = round(confidence, 3),
            method     = "rule_based",
            rationale  = rationale,
            task_id    = task_id,
        )
 
    def decompose_demo_task(self) -> DecompositionResult:
        """
        Decompose the Week 10 demo task exactly as the plan specifies:
        'Build a web scraper, write a research hypothesis about what
        you will find, and generate a diagram of the data flow.'
 
        This is the canonical hard task for PANTHEON. Used in the
        Week 10 demo and as the benchmark for coalition vs solo comparison.
        """
        demo_task = (
            "Build a Python web scraper for Hacker News front page. "
            "Write a hypothesis about what topics dominate today. "
            "Generate a data flow diagram of the scraper architecture."
        )
        tid = str(uuid.uuid4())
        return self.decompose(demo_task, tid, difficulty=0.85)

    def _match_patterns(self, text : str,
                        patterns  : list) -> tuple[str, float]:
        """
        Find the best matching pattern for the given text.
 
        Returns (description_template, confidence).
        confidence = fraction of keywords in best pattern that appear in text.
        Returns ("", 0.0) if no keywords matched at all.
        """
        best_desc = ""
        best_conf = 0.0
 
        for keywords, template in patterns:
            hits = sum(1 for kw in keywords if kw in text)
            if hits == 0:
                continue
            conf = hits / len(keywords)
            if conf > best_conf:
                best_conf = conf
                best_desc = template
 
        return best_desc, best_conf
 
    def _make_subtask(self, parent_id   : str,
                      task_type   : SubtaskType,
                      description : str,
                      difficulty  : float) -> Subtask:
        """Construct a Subtask with a fresh UUID."""
        return Subtask(
            subtask_id  = str(uuid.uuid4()),
            parent_id   = parent_id,
            task_type   = task_type,
            description = description,
            difficulty  = difficulty,
        )

 
    def decompose_llm(self, task_description : str,
                      task_id        : str,
                      difficulty     : float = 0.7,
                      llm_fn         = None) -> DecompositionResult:
        """
        LLM-based decomposition using Llama-3.2-3B-Instruct.
 
        llm_fn: callable(prompt: str) → str
          Provided by HistorianAgent in Week 8.
          If None, falls back to rule-based.
 
        Prompt format:
          "Decompose this task into 2-4 subtasks of types
           code/research/visual. Return JSON only:
           [{type, description, difficulty}, ...]"
 
        JSON is parsed and Subtask objects are constructed.
        On parse failure: falls back to rule_based silently.
        """
        if llm_fn is None:
            return self.decompose(task_description, task_id, difficulty)
 
        prompt = (
            f"Decompose this task into 2-4 subtasks. "
            f"Each subtask must have type (code|research|visual), "
            f"a specific description, and a difficulty (0.0-1.0). "
            f"Return ONLY a JSON array, no other text.\n\n"
            f"Task: {task_description}\n\n"
            f"JSON:"
        )
 
        try:
            import json
            raw      = llm_fn(prompt)
            parsed   = json.loads(raw)
            subtasks = []
            for item in parsed[:self.max_subtasks]:
                t = SubtaskType(item["type"])
                subtasks.append(self._make_subtask(
                    parent_id   = task_id,
                    task_type   = t,
                    description = item["description"],
                    difficulty  = float(item.get("difficulty", difficulty)),
                ))
            return DecompositionResult(
                subtasks   = subtasks,
                confidence = 0.85,
                method     = "llm",
                rationale  = "LLM decomposition",
                task_id    = task_id,
            )
        except Exception as e:
           
            result          = self.decompose(task_description, task_id, difficulty)
            result.rationale = f"LLM failed ({e}); used rule_based"
            return result

 
if __name__ == "__main__":
    print("=" * 56)
    print("PANTHEON Week 4 — TaskDecomposer smoke test")
    print("=" * 56)
 
    decomposer = TaskDecomposer(min_subtasks=2, max_subtasks=5,
                                min_types=2)

    print("\n── Test 1: Week 10 demo task ──")
    result = decomposer.decompose_demo_task()
    print(result)
    print(f"confidence : {result.confidence}")
    print(f"rationale  : {result.rationale}")
    print(f"method     : {result.method}")
 
    types_found = {s.task_type for s in result.subtasks}
    assert SubtaskType.CODE     in types_found, "Missing code subtask"
    assert SubtaskType.RESEARCH in types_found, "Missing research subtask"
    assert SubtaskType.VISUAL   in types_found, "Missing visual subtask"
    assert 2 <= len(result.subtasks) <= 5
    print(" demo task: all three types present")

    print("\n── Test 2: code-heavy task ──")
    r2 = decomposer.decompose(
        "implement a binary search algorithm and write tests for it",
        str(uuid.uuid4()), difficulty=0.6
    )
    print(r2)
    assert SubtaskType.CODE in {s.task_type for s in r2.subtasks}
    assert len(r2.subtasks) >= 2   # min_types=2 forces a second type
    print(f" code task: {len(r2.subtasks)} subtasks, "
          f"types={[s.task_type.value for s in r2.subtasks]}")

    print("\n── Test 3: research-heavy task ──")
    r3 = decomposer.decompose(
        "write a hypothesis about why agent specialization emerges",
        str(uuid.uuid4()), difficulty=0.5
    )
    print(r3)
    assert SubtaskType.RESEARCH in {s.task_type for s in r3.subtasks}
    print(f" research task: {len(r3.subtasks)} subtasks")

    print("\n── Test 4: ambiguous task (fallbacks) ──")
    r4 = decomposer.decompose(
        "do something interesting with the society data",
        str(uuid.uuid4()), difficulty=0.4
    )
    print(r4)
    assert len(r4.subtasks) >= 2
    assert r4.confidence < 0.7   
    print(f" ambiguous task: conf={r4.confidence}, "
          f"fallbacks used as expected")
 
    print("\n── Test 5: difficulty scaling ──")
    r5 = decomposer.decompose(
        "build a graph traversal algorithm and visualize it",
        str(uuid.uuid4()), difficulty=0.9
    )
    for s in r5.subtasks:
        assert s.difficulty <= 0.9 + 0.01, \
            f"Subtask difficulty {s.difficulty} exceeds parent"
        assert s.reward == max(1, int(s.difficulty * 20))
    print(f" difficulty scaling: "
          f"{[round(s.difficulty,2) for s in r5.subtasks]}")
    print(f"  rewards: {[s.reward for s in r5.subtasks]}")
 
    print("\n── Test 6: serialization ──")
    for s in result.subtasks:
        d = s.to_dict()
        assert all(k in d for k in
                   ["subtask_id", "parent_id", "task_type",
                    "description", "difficulty", "reward"])
    print(" all subtasks serialize to dict correctly")
 
    print("\n" + "=" * 56)
    print("TaskDecomposer — DONE")
    print("Next: coalition/auction.py")
    print("=" * 56)
 


