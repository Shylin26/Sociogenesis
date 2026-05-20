import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import random
import numpy as np
import torch
import torch.nn.functional as F

from output.output_router      import OutputRouter
from speciation.engine         import SpeciationEngine
from substrate.economy         import ComputeEconomy
from substrate.registry        import AgentRegistry
from substrate.artifact_store  import ArtifactStore, ArtifactType
from coalition.task_decomposer import TaskDecomposer
from coalition.Auction         import AuctionEngine
from coalition.coalition       import CoalitionFormation
from coalition.aggregator      import CoalitionAggregator
from memory.episodic           import EpisodicMemory
from memory.distillation       import KnowledgeDistiller
from memory.librarian          import LibrarianAgent

BENCHMARK_PROBLEMS = [
    ("code",     "Write a Python function to reverse a linked list.",           0.30, 10),
    ("code",     "Write a Python bubble sort implementation.",                  0.40, 10),
    ("code",     "Write a Python binary search function.",                      0.35, 10),
    ("research", "Explain why insertion sort beats quicksort on small arrays.", 0.50, 12),
    ("research", "Describe the tradeoff between precision and recall.",         0.45, 12),
    ("code",     "Write a Python web scraper for Hacker News.",                 0.85, 20),
    ("code",     "Implement a thread-safe LRU cache in Python.",                0.80, 20),
    ("code",     "Write a Python async HTTP client with retry logic.",          0.78, 20),
    ("research", "Compare quicksort vs mergesort empirically.",                 0.60, 16),
    ("research", "Analyse time complexity of Dijkstra vs Bellman-Ford.",        0.65, 16),
    ("research", "Evaluate transformer attention vs linear attention.",         0.70, 16),
    ("visual",   "Draw a t-SNE plot of agent fingerprints.",                    0.60, 12),
    ("visual",   "Generate a bar chart of token balances across agents.",       0.55, 12),
    ("research", "Describe emergent specialisation in multi-agent systems.",    0.68, 16),
    ("code",     "Implement a min-heap in Python from scratch.",                0.77, 20),
    ("visual",   "Draw a force-directed graph of coalition formations.",        0.88, 18),
    ("visual",   "Render a heatmap of agent skill fingerprints.",               0.82, 18),
    ("visual",   "Visualise knowledge graph node growth over ticks.",           0.79, 18),
    ("research", "Propose a falsifiable hypothesis on RAG recall improvement.", 0.85, 20),
    ("code",     "Write a Python actor-model message passing prototype.",       0.90, 22),
]

COALITION_DIFFICULTY_THRESHOLD = 0.75
N_AGENTS                       = 10


def _build_fingerprints(engine):
    return {aid: rec.fingerprint for aid, rec in engine.records.items()}


def _build_type_seeds(engine):
    return dict(engine._type_seeds)


def _build_reputations(registry):
    return {aid: rec.reputation_score for aid, rec in registry.records.items()}


def _build_private_latents(n_agents, seed=99):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return {
        i: F.normalize(torch.randn(128, generator=gen), dim=0)
        for i in range(n_agents)
    }


def _score_problem(task_type, task_desc, difficulty, reward, tick,
                   engine, router, cf, aggregator, decomposer,
                   economy, registry, private_latents, living):
    fingerprints = _build_fingerprints(engine)
    type_seeds   = _build_type_seeds(engine)
    balances     = economy.snapshot()
    reputations  = _build_reputations(registry)
    task_id      = str(uuid.uuid4())

    if difficulty >= COALITION_DIFFICULTY_THRESHOLD:
        decomp  = decomposer.decompose(task_desc, task_id, difficulty=difficulty)
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
                economy.spend(r.winner_id, r.tokens_spent)

        coalition = cf.form(
            parent_task_id  = task_id,
            auction_results = results,
            subtasks        = decomp.subtasks,
            reputations     = reputations,
            private_latents = private_latents,
            token_balances  = economy.snapshot(),
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
            type_map = {
                "code"    : routed.code_artifact,
                "research": routed.research_artifact,
                "visual"  : routed.visual_artifact,
            }
            for member in coalition.members:
                art = type_map.get(member.subtask_type)
                if art:
                    content = art.code if member.subtask_type == "code" \
                              else art.raw_text if member.subtask_type == "research" \
                              else art.content
                    cf.record_output(coalition.coalition_id, member.agent_id,
                                     content, art.quality_score, tick)

            agg_out   = aggregator.aggregate(coalition)
            completed = cf.complete(coalition.coalition_id, agg_out.content, tick)
            for member in completed.members:
                if member.reward_share > 0:
                    economy.earn(member.agent_id, member.reward_share)
                registry.update_reputation(member.agent_id, member.quality_score)
                engine.update_fingerprint(
                    agent_id  = member.agent_id,
                    task_type = member.subtask_type,
                    success   = member.quality_score >= 0.5,
                    reward    = member.reward_share,
                    tick      = tick,
                )
            return routed.mean_quality
        return 0.0

    else:
        best_agent = engine.route_task_by_seed(task_type, living)
        if task_type == "code":
            art     = router.code_layer.produce(agent_id=best_agent,
                                                task_desc=task_desc, tick=tick,
                                                difficulty=difficulty)
            quality = art.quality_score
        elif task_type == "research":
            art     = router.research_layer.produce(agent_id=best_agent,
                                                    task_desc=task_desc, tick=tick,
                                                    difficulty=difficulty)
            quality = art.quality_score
        else:
            art     = router.visual_layer.produce(agent_id=best_agent,
                                                  task_desc=task_desc, tick=tick,
                                                  difficulty=difficulty)
            quality = art.quality_score

        if quality >= 0.5:
            economy.earn(best_agent, reward)
        else:
            economy.spend(best_agent, max(1, int(difficulty * 7)))
        engine.update_fingerprint(
            agent_id  = best_agent,
            task_type = task_type,
            success   = quality >= 0.5,
            reward    = reward if quality >= 0.5 else 0,
            tick      = tick,
        )
        registry.update_reputation(best_agent, quality)
        return quality


def run_benchmark(engine, router, cf, aggregator, decomposer,
                  economy, registry, private_latents, librarian,
                  tick_offset=0) -> float:
    living = list(range(N_AGENTS))
    scores = []
    for i, (task_type, task_desc, difficulty, reward) in enumerate(BENCHMARK_PROBLEMS):
        tick = tick_offset + i
        q    = _score_problem(task_type, task_desc, difficulty, reward, tick,
                              engine, router, cf, aggregator, decomposer,
                              economy, registry, private_latents, living)
        scores.append(q)
        librarian.on_tick(tick)
    return float(np.mean(scores))


def run_benchmark_harness(n_runs=5, ticks_between_runs=500, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    economy        = ComputeEconomy(N_AGENTS)
    registry       = AgentRegistry(N_AGENTS)
    engine         = SpeciationEngine(n_agents=N_AGENTS, alpha_scale=0.15,
                                      top_n=3, seed=seed)
    router         = OutputRouter(visual_mode="ascii")
    cf             = CoalitionFormation()
    aggregator     = CoalitionAggregator()
    decomposer     = TaskDecomposer(min_types=3)
    private_latents= _build_private_latents(N_AGENTS, seed=99)
    episodic       = EpisodicMemory(db_path="./artifacts/benchmark_episodic.db")
    distiller      = KnowledgeDistiller(episodic,
                                        db_path="./artifacts/benchmark_distill.db")
    librarian      = LibrarianAgent.seed(episodic, distiller)

    improvement_curve = []
    tick_offset       = 0

    for run in range(1, n_runs + 1):
        mean_q = run_benchmark(engine, router, cf, aggregator, decomposer,
                               economy, registry, private_latents, librarian,
                               tick_offset=tick_offset)
        improvement_curve.append(round(mean_q, 4))
        tick_offset += ticks_between_runs

    run1       = improvement_curve[0]
    run_last   = improvement_curve[-1]
    improvement= (run_last - run1) / (run1 + 1e-8)

    return improvement_curve, improvement


if __name__ == "__main__":
    curve, improvement = run_benchmark_harness(n_runs=5, ticks_between_runs=500)

    print("\n=== BENCHMARK HARNESS ===")
    for i, score in enumerate(curve, 1):
        print(f"  run {i}: mean_quality={score:.4f}")
    print(f"\n  improvement run1→run{len(curve)}: {improvement*100:.1f}%")
    passed = improvement >= 0.30
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} (threshold=30%)")
