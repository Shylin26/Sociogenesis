import sys
import os
import random
import uuid
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coalition.task_decomposer import TaskDecomposer
from coalition.Auction         import AuctionEngine
from coalition.coalition       import CoalitionFormation
from coalition.aggregator      import CoalitionAggregator
from output.output_router      import OutputRouter
from speciation.engine         import SpeciationEngine
from substrate.economy         import ComputeEconomy
from substrate.registry        import AgentRegistry
from substrate.artifact_store  import ArtifactStore
from memory.episodic           import EpisodicMemory, DIM
from memory.distillation       import KnowledgeDistiller
from memory.librarian          import LibrarianAgent

TASK_BANK = [
    ("code",     "Write a Python function to reverse a linked list.",            0.30, 10),
    ("code",     "Write a Python bubble sort implementation.",                   0.40, 10),
    ("code",     "Write a Python binary search function.",                       0.35, 10),
    ("research", "Explain why insertion sort beats quicksort on small arrays.",  0.50, 12),
    ("research", "Describe the tradeoff between precision and recall.",          0.45, 12),
    ("code",     "Write a Python web scraper for Hacker News.",                  0.85, 20),
    ("code",     "Implement a thread-safe LRU cache in Python.",                 0.80, 20),
    ("code",     "Write a Python async HTTP client with retry logic.",           0.78, 20),
    ("research", "Compare quicksort vs mergesort empirically.",                  0.60, 16),
    ("research", "Analyse time complexity of Dijkstra vs Bellman-Ford.",         0.65, 16),
    ("research", "Evaluate transformer attention vs linear attention.",          0.70, 16),
    ("visual",   "Draw a t-SNE plot of agent fingerprints.",                     0.60, 12),
    ("visual",   "Generate a bar chart of token balances across agents.",        0.55, 12),
    ("research", "Describe emergent specialisation in multi-agent systems.",     0.68, 16),
    ("code",     "Implement a min-heap in Python from scratch.",                 0.77, 20),
    ("visual",   "Draw a force-directed graph of coalition formations.",         0.88, 18),
    ("visual",   "Render a heatmap of agent skill fingerprints.",                0.82, 18),
    ("visual",   "Visualise knowledge graph node growth over ticks.",            0.79, 18),
    ("research", "Propose a falsifiable hypothesis on RAG recall improvement.",  0.85, 20),
    ("code",     "Write a Python actor-model message passing prototype.",        0.90, 22),
]

FIXED_ORDER         = list(range(len(TASK_BANK)))
COALITION_THRESHOLD = 0.75
N_AGENTS            = 10
TASK_TYPES          = ["code", "research", "visual"]


def _make_type_seeds(seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return {
        name: F.normalize(torch.randn(128, generator=gen), dim=0)
        for name in TASK_TYPES
    }


def _make_fingerprints(type_seeds, seed=7):
    gen = torch.Generator()
    gen.manual_seed(seed)
    fps = {}
    for i in range(N_AGENTS):
        base = (type_seeds["code"]     if i < 3 else
                type_seeds["research"] if i < 6 else
                type_seeds["visual"])
        fps[i] = F.normalize(base + torch.randn(128, generator=gen) * 0.1, dim=0)
    return fps


def _make_private_latents(seed=99):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return {
        i: F.normalize(torch.randn(128, generator=gen), dim=0)
        for i in range(N_AGENTS)
    }


def _embed(seed_int):
    rng = np.random.RandomState(seed_int % (2 ** 31))
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v


def _run_one_pass(
    engine, cf, aggregator, router,
    episodic, librarian,
    type_seeds, fingerprints, private_latents,
    living, tick_offset,
):
    scores      = []
    balances    = {i: 100  for i in range(N_AGENTS)}
    reputations = {i: 0.0  for i in range(N_AGENTS)}

    for i in FIXED_ORDER:
        task_type, task_desc, difficulty, reward = TASK_BANK[i]
        task_id     = str(uuid.uuid4())
        tick        = tick_offset + i
        task_emb    = _embed(i)
        rag_context = episodic.retrieve(task_emb, k=3)

        if difficulty >= COALITION_THRESHOLD:
            decomp  = TaskDecomposer(min_types=3).decompose(
                task_desc, task_id, difficulty=difficulty
            )
            ae      = AuctionEngine()
            results = ae.run_all_auctions(
                subtasks       = decomp.subtasks,
                living_agents  = living,
                fingerprints   = fingerprints,
                token_balances = balances,
                type_seeds     = type_seeds,
            )
            for r in results:
                if r.has_winner:
                    balances[r.winner_id] = max(
                        0, balances[r.winner_id] - r.tokens_spent
                    )

            coalition = cf.form(
                parent_task_id  = task_id,
                auction_results = results,
                subtasks        = decomp.subtasks,
                reputations     = reputations,
                private_latents = private_latents,
                token_balances  = balances,
                tick            = tick,
            )

            if coalition is not None:
                routed = router.route_and_produce(
                    coalition    = coalition,
                    task_desc    = task_desc,
                    tick         = tick,
                    fingerprints = fingerprints,
                    type_seeds   = type_seeds,
                    difficulty   = difficulty,
                )
                quality = routed.mean_quality

                type_map = {
                    "code"    : routed.code_artifact,
                    "research": routed.research_artifact,
                    "visual"  : routed.visual_artifact,
                }
                for member in coalition.members:
                    art = type_map.get(member.subtask_type)
                    if art:
                        cf.record_output(
                            coalition.coalition_id,
                            member.agent_id,
                            art.quality_score,
                            tick,
                        )

                agg_out   = aggregator.aggregate(coalition)
                completed = cf.complete(coalition.coalition_id, agg_out.content, tick)

                for member in completed.members:
                    if member.reward_share > 0:
                        balances[member.agent_id] += member.reward_share
                    reputations[member.agent_id] = (
                        0.9 * reputations[member.agent_id]
                        + 0.1 * member.quality_score
                    )

                solution_emb = _embed(hash(agg_out.content) % (2 ** 31))
                episodic.record(
                    task_emb     = task_emb,
                    solution_emb = solution_emb,
                    quality      = quality,
                    task_type    = task_type,
                    agent_ids    = [m.agent_id for m in completed.members],
                    tick         = tick,
                )
            else:
                quality = 0.0

        else:
            best_agent = engine.route_task_by_seed(task_type, living)

            if task_type == "code":
                art     = router.code_layer.produce(
                    agent_id=best_agent, task_desc=task_desc,
                    tick=tick, difficulty=difficulty,
                )
                quality = art.quality_score
                content = art.code

            elif task_type == "research":
                art     = router.research_layer.produce(
                    agent_id=best_agent, task_desc=task_desc,
                    tick=tick, difficulty=difficulty,
                )
                quality = art.quality_score
                content = art.raw_text

            else:
                art     = router.visual_layer.produce(
                    agent_id=best_agent, task_desc=task_desc,
                    tick=tick, difficulty=difficulty,
                )
                quality = art.quality_score
                content = art.content

            if quality >= 0.5:
                balances[best_agent] += reward
            else:
                balances[best_agent] = max(
                    0, balances[best_agent] - max(1, int(difficulty * 7))
                )

            reputations[best_agent] = (
                0.9 * reputations[best_agent] + 0.1 * quality
            )

            solution_emb = _embed(hash(content) % (2 ** 31))
            episodic.record(
                task_emb     = task_emb,
                solution_emb = solution_emb,
                quality      = quality,
                task_type    = task_type,
                agent_ids    = [best_agent],
                tick         = tick,
            )

        scores.append(quality)
        librarian.on_tick(tick)

    return float(np.mean(scores))


def run_benchmark_harness(n_runs=5, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    type_seeds      = _make_type_seeds(seed=seed)
    fingerprints    = _make_fingerprints(type_seeds, seed=7)
    private_latents = _make_private_latents(seed=99)
    living          = list(range(N_AGENTS))

    engine     = SpeciationEngine(n_agents=N_AGENTS, alpha_scale=0.15,
                                  top_n=3, seed=seed)
    cf         = CoalitionFormation()
    aggregator = CoalitionAggregator()
    router     = OutputRouter(visual_mode="ascii")
    episodic   = EpisodicMemory()
    distiller  = KnowledgeDistiller(episodic=episodic)
    librarian  = LibrarianAgent(episodic=episodic, distiller=distiller)

    improvement_curve = []
    tick_offset       = 0

    for run in range(1, n_runs + 1):
        mean_q = _run_one_pass(
            engine, cf, aggregator, router,
            episodic, librarian,
            type_seeds, fingerprints, private_latents,
            living, tick_offset,
        )
        improvement_curve.append(mean_q)
        tick_offset += len(TASK_BANK)

    run1        = improvement_curve[0]
    run5        = improvement_curve[-1]
    improvement = (run5 - run1) / (run1 + 1e-8)

    return improvement_curve, improvement


if __name__ == "__main__":
    curve, improvement = run_benchmark_harness(n_runs=5, seed=42)

    print("\n=== BENCHMARK HARNESS SMOKE TEST ===")
    for i, score in enumerate(curve, 1):
        print(f"  run {i}: mean_quality={score:.4f}")
    print(f"\n  improvement run1→run5: {improvement * 100:.1f}%")
    passed = improvement >= 0.30
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (threshold=30%)")