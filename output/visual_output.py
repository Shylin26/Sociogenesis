import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import json
import re
from dataclasses import dataclass,field
from typing import Optional

@dataclass
class VisualArtifact:
    artifact_id   : str
    agent_id      : int
    prompt        : str
    style_vector  : list
    content       : str
    mode          : str
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
            "prompt"        : self.prompt[:100],
            "mode"          : self.mode,
            "content_length": len(self.content),
        }


def make_data_flow_diagram(components: list[str],
                           title: str = "Data Flow") -> str:
    max_len  = max((len(c) for c in components), default=20)
    box_w    = max(25, max_len + 4)
    pad      = box_w - 2
    lines    = []
    tlen     = len(title) + 2
    lines.append(f"  ┌{'─' * tlen}┐")
    lines.append(f"  │ {title} │")
    lines.append(f"  └{'─' * tlen}┘")
    lines.append(f"       │")
    lines.append(f"       ▼")
    for i, comp in enumerate(components):
        label = comp[:pad-2]
        lines.append(f"  ┌{'─' * pad}┐")
        lines.append(f"  │  {label:<{pad-2}}│")
        lines.append(f"  └{'─' * pad}┘")
        if i < len(components) - 1:
            lines.append(f"       │")
            lines.append(f"       ▼")
    return "\n".join(lines)


def extract_components(task_desc: str) -> tuple[list[str], str]:
    desc = task_desc.lower()
    if any(w in desc for w in ["scraper", "scrape", "hacker news", "crawl", "fetch"]):
        return ([
            "HTTP GET Request",
            "HTML Response Parser",
            "Title & Score Extractor",
            "Topic Classifier",
            "Results [ ]",
            "Output / Storage",
        ], "Web Scraper Architecture")
    if any(w in desc for w in ["agent", "coalition", "speciali", "society"]):
        return ([
            "Task Posted",
            "Auction Engine",
            "Coalition Formation",
            "Specialist Agents",
            "Output Aggregator",
            "Artifact Store",
        ], "Agent Society Flow")
    if any(w in desc for w in ["memory", "episodic", "distill", "rag"]):
        return ([
            "Task + Solution",
            "EpisodicMemory (FAISS)",
            "KnowledgeDistiller",
            "Semantic Graph",
            "RAG Retrieval",
            "Context Injection",
        ], "Memory Pipeline")
    if any(w in desc for w in ["train", "model", "neural", "transform"]):
        return ([
            "Input Tokens",
            "Embedding Layer",
            "Transformer Blocks",
            "Output Projection",
            "Loss Computation",
            "Gradient Update",
        ], "Training Pipeline")
    words = [w.capitalize() for w in task_desc.split() if len(w) > 4][:6]
    return (words or ["Input", "Process", "Output"], "Data Flow")
 
 
def make_bar_chart(data: dict[str, float], title: str = "Chart") -> str:
    """Generate an ASCII bar chart."""
    max_val  = max(data.values()) if data else 1
    max_bar  = 30
    lines    = [f"\n  {title}", f"  {'─' * (max_bar + 20)}"]
    for label, val in sorted(data.items(), key=lambda x: x[1], reverse=True):
        bar_len  = int((val / max_val) * max_bar)
        bar      = "█" * bar_len
        lines.append(f"  {label:<15s} │{bar:<30s} {val:.2f}")
    lines.append(f"  {'─' * (max_bar + 20)}")
    return "\n".join(lines)
 
 
def make_network_graph(nodes: list[str],
                        edges: list[tuple],
                        title: str = "Network") -> str:
    """Generate an ASCII network graph description."""
    lines = [f"\n  {title}  (force-directed layout)", ""]
    lines.append("  Nodes:")
    for i, node in enumerate(nodes):
        size = "●" if i < len(nodes) // 3 else "○"
        lines.append(f"    {size} {node}")
    lines.append("\n  Edges:")
    for src, dst in edges:
        lines.append(f"    {src} ──→ {dst}")
    return "\n".join(lines)
 
 
def make_tsne_plot(clusters: dict[str, list[str]],
                   title: str = "t-SNE Projection") -> str:
    """Generate an ASCII t-SNE cluster visualization."""
    symbols = {"code": "▲", "research": "●", "visual": "■"}
    lines   = [f"\n  {title}", f"  {'─' * 40}",
               "  dim₂ ↑"]
 
    
    grid = [[" "] * 20 for _ in range(8)]
 
    import random
    rng = random.Random(42)
    cluster_positions = {
        "code"    : (2, 4),
        "research": (5, 2),
        "visual"  : (4, 6),
    }
 
    for cluster, agents in clusters.items():
        cx, cy = cluster_positions.get(cluster, (3, 3))
        sym    = symbols.get(cluster, "●")
        for agent in agents:
            x = min(19, max(0, cx + rng.randint(-2, 2)))
            y = min(7,  max(0, cy + rng.randint(-1, 1)))
            grid[y][x] = sym
 
    for row in reversed(grid):
        lines.append("  │" + "".join(row) + "│")
    lines.append("  └" + "─" * 20 + "┘→ dim₁")
    lines.append(f"\n  Legend: ▲=code  ●=research  ■=visual")
    return "\n".join(lines)
 
 
def make_timeline(events: list[tuple[int, str]],
                  title: str = "Timeline") -> str:
    """Generate an ASCII timeline."""
    lines = [f"\n  {title}", ""]
    for tick, event in sorted(events):
        lines.append(f"  tick {tick:4d} ──● {event}")
    return "\n".join(lines)

VISUAL_TEMPLATES = {
    "data_flow": {
        "prompt": (
            "Data flow diagram, technical architecture, clean lines, "
            "boxes and arrows, white background, professional diagram style"
        ),
        "style_vector": [0.8, 0.1, 0.1, 0.9, 0.2],
        "generator": lambda ctx: make_data_flow_diagram(
            ctx.get("components") or extract_components(ctx.get("task_desc", ctx.get("title","")))[0],
            title=ctx.get("title") or extract_components(ctx.get("task_desc", ctx.get("title","")))[1]
        ),
    },
 
    "bar_chart": {
        "prompt": (
            "Bar chart visualization, data analysis, clean minimal style, "
            "labeled axes, color-coded bars, professional infographic"
        ),
        "style_vector": [0.6, 0.3, 0.2, 0.8, 0.4],
        "generator": lambda ctx: make_bar_chart(
            ctx.get("data", {
                "AI / ML"     : 0.42,
                "Programming" : 0.28,
                "Business"    : 0.15,
                "Science"     : 0.10,
                "Other"       : 0.05,
            }),
            title=ctx.get("title", "Hacker News Topic Distribution")
        ),
    },
 
    "tsne_plot": {
        "prompt": (
            "t-SNE scatter plot, dimensionality reduction visualization, "
            "colored clusters, labeled legend, scientific plot style"
        ),
        "style_vector": [0.7, 0.5, 0.1, 0.9, 0.3],
        "generator": lambda ctx: make_tsne_plot(
            ctx.get("clusters", {
                "code"    : ["A0", "A1", "A2"],
                "research": ["A3", "A4", "A5"],
                "visual"  : ["A6", "A7", "A8", "A9"],
            }),
            title=ctx.get("title", "Agent Fingerprint Clusters")
        ),
    },
 
    "network_graph": {
        "prompt": (
            "Network graph visualization, force-directed layout, "
            "nodes and edges, community detection, graph theory diagram"
        ),
        "style_vector": [0.5, 0.6, 0.2, 0.7, 0.5],
        "generator": lambda ctx: make_network_graph(
            ctx.get("nodes", [
                "Coalition A", "Agent 0", "Agent 3", "Agent 6",
                "Coalition B", "Agent 1", "Agent 4",
            ]),
            ctx.get("edges", [
                ("Coalition A", "Agent 0"),
                ("Coalition A", "Agent 3"),
                ("Coalition A", "Agent 6"),
                ("Coalition B", "Agent 1"),
                ("Coalition B", "Agent 4"),
                ("Agent 0", "Agent 1"),
            ]),
            title=ctx.get("title", "Coalition Network")
        ),
    },
 
    "timeline": {
        "prompt": (
            "Timeline visualization, chronological sequence, "
            "milestone markers, clean horizontal layout, event labels"
        ),
        "style_vector": [0.4, 0.2, 0.5, 0.8, 0.3],
        "generator": lambda ctx: make_timeline(
            ctx.get("events", [
                (0,   "Society initialized — 10 agents, 100 tokens each"),
                (50,  "First specialization clusters emerge"),
                (100, "Coalition formation begins"),
                (200, "Knowledge graph reaches 50 nodes"),
                (300, "Society performance benchmark run"),
                (500, "HistorianAgent writes self-report"),
            ]),
            title=ctx.get("title", "PANTHEON Society Timeline")
        ),
    },
}
KEYWORD_MAP = {
    "flow"      : "data_flow",
    "diagram"   : "data_flow",
    "architect" : "data_flow",
    "scraper"   : "data_flow",
    "pipeline"  : "data_flow",
    "bar chart" : "bar_chart",
    "chart"     : "bar_chart",
    "distribut" : "bar_chart",
    "topic"     : "bar_chart",
    "t-sne"     : "tsne_plot",
    "tsne"      : "tsne_plot",
    "cluster"   : "tsne_plot",
    "fingerprint": "tsne_plot",
    "scatter"   : "tsne_plot",
    "network"   : "network_graph",
    "graph"     : "network_graph",
    "coalition" : "network_graph",
    "timeline"  : "timeline",
    "history"   : "timeline",
    "sequence"  : "timeline",
}

class CLIPScorer:
    STOP_WORDS = {"a", "an", "the", "and", "or", "of", "in",
                  "on", "at", "to", "for", "with", "by", "from"}
 
    def score(self, prompt: str, content: str) -> float:
        prompt_words  = set(re.findall(r'\w+', prompt.lower())) \
                        - self.STOP_WORDS
        content_words = set(re.findall(r'\w+', content.lower()))
        if not prompt_words:
            return 0.5
        overlap = prompt_words & content_words
        raw     = len(overlap) / len(prompt_words)
        structure_bonus = 0.0
        if '┌' in content and '└' in content:
            structure_bonus += 0.15
        if '▼' in content or '│' in content:
            structure_bonus += 0.10
        if content.count('┌') >= 3:
            structure_bonus += 0.10
        return round(max(0.3, min(1.0, raw + 0.3 + structure_bonus)), 3)

class VisualOutputLayer:
    def __init__(self, mode: str = "ascii"):
        assert mode in ("sdxl", "ascii", "stub")
        self.mode    = mode
        self.scorer  = CLIPScorer()
        self.history : list[VisualArtifact] = []
        self.total_produced = 0
        self._sdxl_pipe = None   
 
    def select_template(self, task_desc: str) -> str:
        """Pick best template key for this task description."""
        desc_lower = task_desc.lower()
        for keyword, template_name in KEYWORD_MAP.items():
            if keyword in desc_lower:
                return template_name
        return "data_flow"   
 
    def produce(self, agent_id      : int,
                task_desc     : str,
                tick          : int,
                coalition_id  : Optional[str] = None,
                context       : Optional[dict] = None,
                template_key  : Optional[str]  = None,
                difficulty    : float = 0.5) -> VisualArtifact:
        key      = template_key or self.select_template(task_desc)
        template = VISUAL_TEMPLATES.get(key, VISUAL_TEMPLATES["data_flow"])
        ctx      = dict(context or {})
        ctx.setdefault("task_desc", task_desc)

        prompt       = template["prompt"]
        style_vector = template["style_vector"]

        if self.mode == "sdxl":
            content = self._generate_sdxl(prompt)
        elif self.mode == "ascii":
            content = template["generator"](ctx)
        else:
            content = (
                f"[VISUAL STUB]\n"
                f"prompt: {prompt[:80]}\n"
                f"style:  {style_vector}\n"
                f"task:   {task_desc[:80]}\n"
            )

        import random
        base_quality = self.scorer.score(task_desc, content)
        if difficulty < 0.4:
            quality = base_quality * random.uniform(0.4, 0.65)
        elif difficulty < 0.75:
            quality = base_quality * random.uniform(0.65, 0.90)
        else:
            quality = base_quality
        quality = round(min(1.0, max(0.1, quality)), 3)
 
        artifact = VisualArtifact(
            artifact_id   = str(uuid.uuid4()),
            agent_id      = agent_id,
            prompt        = prompt,
            style_vector  = style_vector,
            content       = content,
            mode          = self.mode,
            quality_score = quality,
            tick          = tick,
            coalition_id  = coalition_id,
        )
 
        self.history.append(artifact)
        self.total_produced += 1
        return artifact
    
    def _generate_sdxl(self, prompt: str) -> str:
        try:
            if self._sdxl_pipe is None:
                from diffusers import AutoPipelineForText2Image
                import torch
                self._sdxl_pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo",
                    torch_dtype = torch.float16,
                    variant     = "fp16",
                )
                self._sdxl_pipe.to("mps")   
 
            import torch, uuid as _uuid
            result = self._sdxl_pipe(
                prompt          = prompt,
                num_inference_steps = 4,
                guidance_scale  = 0.0,
            )
            img      = result.images[0]
            path     = f"/tmp/pantheon_visual_{_uuid.uuid4().hex[:8]}.png"
            img.save(path)
            return f"[SDXL IMAGE SAVED: {path}]\nprompt: {prompt}"
 
        except ImportError:
            
            template = VISUAL_TEMPLATES["data_flow"]
            return template["generator"]({})
        except Exception as e:
            return f"[SDXL ERROR: {e}]\nprompt: {prompt}"
 
    def snapshot(self) -> dict:
        if not self.history:
            return {"total_produced": 0, "mean_quality": 0.0}
        mean_q = sum(a.quality_score for a in self.history) / len(self.history)
        return {
            "total_produced" : self.total_produced,
            "mean_quality"   : round(mean_q, 3),
            "mode"           : self.mode,
        }

if __name__ == "__main__":
    print("=" * 56)
    print("PANTHEON Week 5 — VisualOutputLayer smoke test")
    print("=" * 56)
 
    layer = VisualOutputLayer(mode="ascii")

    print("\n── Test 1: data flow diagram ──")
    art = layer.produce(
        agent_id  = 8,
        task_desc = "generate a data flow diagram of the scraper architecture",
        tick      = 10,
    )
    print(art.content)
    print(f"\n  quality  : {art.quality_score}")
    print(f"  mode     : {art.mode}")
    assert art.quality_score >= 0.3
    assert len(art.content) > 50
    print("data flow diagram produced")

    print("\n── Test 2: bar chart ──")
    art2 = layer.produce(
        agent_id  = 7,
        task_desc = "create a bar chart of topic distribution",
        tick      = 11,
        context   = {"title": "HN Topic Breakdown"},
    )
    print(art2.content)
    print(f"\n  quality  : {art2.quality_score}")
    assert art2.quality_score >= 0.3
    print("bar chart produced")

    print("\n── Test 3: t-SNE cluster plot ──")
    art3 = layer.produce(
        agent_id  = 6,
        task_desc = "generate a t-SNE plot of agent fingerprint clusters",
        tick      = 12,
    )
    print(art3.content)
    print(f"\n  quality  : {art3.quality_score}")
    assert art3.quality_score >= 0.3
    print("t-SNE plot produced")

    print("\n── Test 4: all templates ──")
    for key in VISUAL_TEMPLATES:
        a = layer.produce(
            agent_id     = 6,
            task_desc    = f"generate {key} visualization",
            tick         = 20,
            template_key = key,
        )
        print(f"   {key:20s} quality={a.quality_score}  "
              f"len={len(a.content)}")
        assert a.quality_score >= 0.3
        assert len(a.content) > 20

    print("\n── Test 5: stub mode ──")
    stub_layer = VisualOutputLayer(mode="stub")
    art5 = stub_layer.produce(
        agent_id  = 9,
        task_desc = "visualize coalition network graph",
        tick      = 30,
    )
    print(art5.content)
    assert "VISUAL STUB" in art5.content
    print("stub mode works")

    print("\n── Test 6: cross-pollination ──")
    flow_art = layer.produce(
        agent_id  = 8,
        task_desc = "data flow diagram of scrape_hackernews function",
        tick      = 40,
        context   = {
            "title"     : "scrape_hackernews Architecture",
            "components": [
                "scrape_hackernews(url)",
                "requests.get(url)",
                "BeautifulSoup parser",
                "results list",
                "classify_topic()",
            ],
        },
    )
    has_ref = "scrape_hackernews" in flow_art.content
    print(f"  function name in diagram: {has_ref}")
    assert has_ref
    print("cross-pollination: code function name appears in visual")
 
    print(f"\nSnapshot: {layer.snapshot()}")
    print("\n" + "=" * 56)
    print("VisualOutputLayer — DONE")
    
    print("=" * 56)
 





