# SOCIOGENESIS

**A self-organizing multi-agent cognitive society that thinks, builds, and evolves — running entirely on a MacBook.**

---

## The Claim

Twenty small transformer agents, given only a hard problem and a shared economy, will:

- Self-organize into specialized roles — without any hardcoded assignment
- Form economic coalitions to solve tasks no single agent can handle alone
- Generate working code, research hypotheses, and visual diagrams
- Distill collective experience into a shared knowledge graph
- Evolve — weak agents die, strong agents reproduce
- Write a structured research paper about their own behavior

No GPUs. No cloud. No LangChain. No prompt pipelines.  
A MacBook M4. From scratch.

---

## Why This Is Different

Every multi-agent system today assigns fixed roles: planner → coder → critic. That is a pipeline, not intelligence.

SOCIOGENESIS agents start **identical**. Same architecture. Same weights. Same token balance. Roles emerge through three forces:

| Force | Mechanism |
|---|---|
| **Task exposure** | Agents that repeatedly succeed at code tasks develop a code-biased skill fingerprint |
| **Reputation routing** | Other agents preferentially bid on tasks that match their fingerprint |
| **Economic pressure** | Agents that fail to earn compute tokens die and are replaced by offspring of successful agents |

This is evolutionary pressure applied to cognition. The result is a computational society whose collective intelligence exceeds the sum of its parts.

---

## Architecture

```
SOCIOGENESIS/
│
├── substrate/              # Economy, registry, artifact store
│   ├── economy.py          # Compute-token market — birth, death, inheritance
│   ├── registry.py         # Global agent state tracker
│   └── artifact_store.py
│
├── agent/                  # Individual agent transformer
│   └── transformer.py      # GPT-2 nano, 3M params, RoPE, RMSNorm, private latent
│
├── speciation/             # Role emergence without supervision
│   ├── engine.py           # 128-dim fingerprint update, attractor convergence
│   └── evolution.py        # Bottom-10% death, top-10% inheritance + mutation
│
├── coalition/              # Multi-agent coordination
│   ├── Auction.py          # Sealed-bid auction — specialists bid cheap
│   ├── coalition.py        # Contract formation, coordinator election
│   ├── task_decomposer.py
│   └── aggregator.py       # Output merging + cross-pollination scoring
│
├── output/                 # Artifact generation
│   ├── code_output.py      # Sandboxed Python execution, partial scoring
│   ├── research_output.py  # Structured hypothesis generation
│   ├── visual_output.py    # ASCII diagram synthesis
│   └── output_router.py    # Specialist assignment by fingerprint similarity
│
├── memory/                 # Society-wide knowledge infrastructure
│   ├── episodic.py         # FAISS-indexed task-solution pairs, Hebbian decay
│   ├── distillation.py     # K-means compression → semantic knowledge graph
│   ├── librarian.py        # Background distillation agent
│   ├── historian.py        # Structured self-reporting, bottleneck detection
│   └── paper_writer.py     # ReportLab PDF generation from run logs
│
├── core/                   # Coordination infrastructure
│   ├── society_model.py    # Online-trained transformer, predicts next event
│   ├── benchmark.py        # Fixed 20-problem harness
│   └── communication_bus.py
│
├── dashboard/              # Live visualization
│   ├── backend.py          # FastAPI + WebSocket, 10Hz stream
│   └── frontend.html       # D3 force graph, live economy, artifact feed
│
├── run_demo.py             # Single-command demo
└── install.sh              # One-script setup
```

---

## System Properties

| Property | SOCIOGENESIS | Current SOTA |
|---|---|---|
| Role assignment | Emergent from task exposure | Hardcoded strings |
| Memory | Shared episodic + semantic graph | Context window only |
| Agent lifecycle | Birth → speciation → death | Static, never dies |
| Coordination | Compute-token auction market | Fixed pipelines |
| Self-reflection | Society models its own behavior | None |
| Output | Code + research + visuals | Usually text only |
| Learning signal | Improves from own history | None between runs |
| Hardware | M4 MacBook, no GPU | Datacenter assumed |

---

## Key Mechanisms

### Skill Fingerprints

Each agent maintains a 128-dimensional vector that drifts toward the centroid of tasks it wins. Fingerprints are initialized identically and diverge purely through task outcomes. Role clusters — code, research, visual — emerge from this process without any label supervision. t-SNE of fingerprints after 500 tasks shows 3+ distinct clusters.

### Compute-Token Economy

Every agent starts with 100 tokens. Completing tasks earns tokens. Bidding in auctions costs tokens. Failing costs tokens. Agents below zero are replaced by mutated offspring of the highest earner. This single mechanism drives specialization, routing, and evolution simultaneously.

### Coalition Auction

Hard tasks (difficulty ≥ 0.75) trigger coalition formation:

1. `TaskDecomposer` splits the task into typed subtasks
2. `AuctionEngine` runs sealed-bid auctions — specialists bid cheaply for domain tasks, generalists bid expensively
3. `CoalitionFormation` elects the highest-reputation agent as coordinator
4. Coordinator has read access to all members' private latent states
5. `CoalitionAggregator` merges outputs and scores cross-references between artifacts

Coalition win rate over solo agents: **100%** on hard tasks.

### Retrieval-Augmented Generation

Before any agent tackles a task, it queries the shared `EpisodicMemory` (FAISS-indexed) for similar past solutions. Retrieved solutions are prepended as context. Quality improves as episodic memory accumulates. The `KnowledgeDistiller` runs k-means every 100 ticks to compress episodic records into a semantic graph — concepts become nodes, similarity becomes edges.

### Society Model

A small transformer (same architecture as the agent core) trains online on the stream of society events: `TASK_POSTED`, `BID_WON`, `COALITION_FORMED`, `TASK_SUCCESS`, `AGENT_DIED`, `AGENT_BORN`. It predicts the next event type and outcome probability. Agents whose behavior surprises the model receive a curiosity bonus, preventing local optima.

### Evolutionary Pressure

Every 200 ticks, the bottom 10% of agents by token balance die. Their slots are filled by mutated offspring of the top 10%. Offspring inherit the parent's skill fingerprint with Gaussian noise added. Bidding strategies are also inherited with perturbation. This is gradient-free evolutionary optimization over the space of agent behaviors.

---

## Results

Across a 1000-tick run with 10 agents:

| Metric | Value |
|---|---|
| Coalition win rate | 100% on hard tasks |
| Mean solution quality | 0.802 |
| Role clusters emerged | 3 (without supervision) |
| Episodic records | 919 |
| Semantic nodes distilled | 15 |
| Evolution events | 5 agent replacements |
| Society model train steps | 978 |
| Society model loss | 1.37 |
| Historian self-reports | 10 |
| Paper word count | 725 |

---

## The Hard Problem Demo

```python
HARD_PROBLEM = (
    "Build a Python web scraper for Hacker News front page. "
    "Write a hypothesis about what topics dominate today. "
    "Generate a data flow diagram of the scraper architecture."
)
```

In one run:

- Coalition [A3, A19, A2] formed in **1 tick**
- Code agent produced a working scraper (quality **1.000**)
- Research agent produced a falsifiable hypothesis (quality **1.000**)
- Visual agent produced a data flow diagram (quality **0.426**)
- All three artifacts cross-reference each other
- Paper generated describing the entire run

---

## Quickstart

```bash
git clone https://github.com/Shylin26/Sociogenesis
cd Sociogenesis
bash install.sh
export KMP_DUPLICATE_LIB_OK=TRUE
python3 run_demo.py
```

The demo runs for ~60 seconds, opens the dashboard automatically, and generates a PDF paper in `artifacts/`.

**Requirements**: Python 3.9+, macOS (M1/M2/M3/M4 recommended). No GPU required.

---

## Full Society Run

```bash
# 1000 ticks — generates paper, evolution, historian reports
python3 week8_loop.py

# 1000 ticks with benchmark harness and society model
python3 week7_loop.py

# Fixed 20-problem evaluation harness
python3 core/benchmark.py

# Week 6 memory loop — episodic + RAG verification
python3 memory/memory_loop.py
```

---

## Dashboard

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python3 -m dashboard.backend
```

Open `http://localhost:8000`.

Live view includes:

- **Agent graph** — D3 force simulation, nodes sized by reputation, colored by role cluster
- **Token economy** — real-time bar chart of compute balances
- **Active coalitions** — members, coordinator, task type, formation tick
- **Artifact feed** — code, research, visual outputs with quality scores
- **Society model** — training steps, prediction loss, curiosity bonuses distributed
- **Live event ticker** — coalition formations, evolutions, milestones in plain English
- **Paper download** — one click to `artifacts/demo_paper.pdf`

---

## Theoretical Grounding

| Paper | Role in SOCIOGENESIS |
|---|---|
| Generative Agents — Park et al., Stanford 2023 | Agent memory + behavior architecture |
| CAMEL — Li et al., 2023 | Multi-agent communication protocol |
| nanoGPT — Karpathy 2022 | Agent transformer base |
| NOTEARS — Zheng et al., 2018 | Causal structure in coalition layer |
| Voyager — Wang et al., 2023 | Skill library + lifelong learning |
| EvoPrompting — Chen et al., 2023 | Evolutionary pressure on agent strategies |

---

## Build Log

| Week | Component | Deliverable |
|---|---|---|
| 1–2 | Substrate + Agent Core | Economy stable, 10 agents alive |
| 3 | Speciation Engine | 3+ role clusters without supervision |
| 4 | Coalition Formation | Coalition beats solo on hard tasks |
| 5 | Generative Output | Code runs, research coherent, visual exists |
| 6 | Shared Memory + RAG | Knowledge graph grows, recall improves |
| 7 | Self-Modeling + Evolution | Society model trains online, agents evolve |
| 8 | Historian + Paper | Society writes a structured research paper |
| 9 | Dashboard | Live D3 visualization at 10Hz |
| 10 | Demo + Hardening | One command, reproducible from scratch |

---

## License

MIT

---

*. Runs on a MacBook. No assigned roles. No hardcoded pipelines. Just agents, an economy, and time.*