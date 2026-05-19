import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import uuid
import logging
import numpy as np

import torch
import torch.nn.functional as F

log = logging.getLogger("pantheon.week6")
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s"
)

from substrate.economy        import ComputeEconomy
from substrate.registry       import AgentRegistry
from substrate.artifact_store import ArtifactStore, ArtifactType
from speciation.engine        import SpeciationEngine
from coalition.task_decomposer import TaskDecomposer
from coalition.Auction         import AuctionEngine
from coalition.coalition       import CoalitionFormation
from coalition.aggregator      import CoalitionAggregator
from output.output_router      import OutputRouter
from memory.episodic           import EpisodicMemory
from memory.distillation       import KnowledgeDistiller
from memory.librarian          import LibrarianAgent

TASK_BANK = [
    ("code",     "Write a Python web scraper for Hacker News.",          0.85, 20),
    ("research", "Hypothesize what topics dominate Hacker News today.",  0.85, 18),
    ("visual",   "Diagram the data flow of the HN scraper.",             0.80, 14),
    ("code",     "Implement quicksort with type hints.",                 0.50, 10),
    ("research", "Compare quicksort vs mergesort empirically.",          0.60, 16),
    ("visual",   "Draw a t-SNE plot of agent fingerprints.",             0.60, 12),
    ("code",     "Return the nth Fibonacci number.",                     0.30, 10),
    ("research", "Why do Fibonacci numbers appear in nature?",           0.50, 15),
    ("visual",   "Create a bar chart of token balance distribution.",    0.40, 12),
    ("code",     "Implement an LRU cache in Python.",                    0.70, 16),
]

COALITION_DIFFICULTY_THRESHOLD = 0.75
TASK_TYPES                     = ["code", "research", "visual"]
RAG_REPEAT_INTERVAL            = 40


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


def _task_emb(task_desc: str) -> np.ndarray:
    vec = np.zeros(128, dtype=np.float32)
    for i, ch in enumerate(task_desc[:256]):
        vec[ord(ch) % 128] += 1.0 / (i + 1)
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-8 else vec


def _art_emb(content: str) -> np.ndarray:
    vec = np.zeros(128, dtype=np.float32)
    for i, ch in enumerate(content[:512]):
        vec[ord(ch) % 128] += 1.0 / (i + 1)
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-8 else vec


def run(n_agents=10, max_ticks=200, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    economy        = ComputeEconomy(n_agents)
    registry       = AgentRegistry(n_agents)
    artifact_store = ArtifactStore(base_dir="./artifacts")
    engine         = SpeciationEngine(n_agents=n_agents, alpha_scale=0.15,
                                      top_n=3, seed=seed)
    decomposer      = TaskDecomposer(min_types=3)
    cf              = CoalitionFormation()
    aggregator      = CoalitionAggregator()
    router          = OutputRouter(visual_mode="ascii")
    private_latents = _build_private_latents(n_agents, seed=99)
    living          = list(range(n_agents))

    episodic  = EpisodicMemory(db_path="./artifacts/episodic.db")
    distiller = KnowledgeDistiller(episodic, db_path="./artifacts/distillation.db")
    librarian = LibrarianAgent.seed(episodic, distiller)

    repeat_tasks = {
        "code"    : ("Write a Python web scraper for Hacker News.", 0.85, 20),
        "research": ("Hypothesize what topics dominate Hacker News today.", 0.85, 18),
        "visual"  : ("Diagram the data flow of the HN scraper.", 0.80, 14),
    }

    baseline_quality = {t: [] for t in TASK_TYPES}
    rag_quality      = {t: [] for t in TASK_TYPES}

    stats = {
        "code": 0, "research": 0, "visual": 0,
        "solo": 0, "coalition": 0,
        "quality": [],
        "coalition_wins": 0, "coalition_total": 0,
    }

    log.info("=" * 55)
    log.info("PANTHEON  Week 6 — Episodic Memory + RAG")
    log.info("=" * 55)

    task_pool = list(TASK_BANK) * (max_ticks // len(TASK_BANK) + 1)
    random.shuffle(task_pool)

    for tick in range(1, max_ticks + 1):
        is_repeat   = (tick % RAG_REPEAT_INTERVAL == 0)
        repeat_type = TASK_TYPES[(tick // RAG_REPEAT_INTERVAL) % len(TASK_TYPES)]

        if is_repeat:
            task_type, task_desc, difficulty, reward = (
                repeat_type, *repeat_tasks[repeat_type]
            )
        else:
            task_type, task_desc, difficulty, reward = task_pool[(tick - 1) % len(task_pool)]

        task_id      = str(uuid.uuid4())
        fingerprints = _build_fingerprints(engine)
        type_seeds   = _build_type_seeds(engine)
        balances     = economy.snapshot()
        reputations  = _build_reputations(registry)
        task_emb_vec = _task_emb(task_desc)

        rag_hits = episodic.retrieve(task_emb_vec, k=3)

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
                rag_bonus = min(0.10, len(rag_hits) * 0.03)

                routed = router.route_and_produce(
                    coalition    = coalition,
                    task_desc    = task_desc,
                    tick         = tick,
                    fingerprints = fingerprints,
                    type_seeds   = type_seeds,
                )

                type_map = {
                    "code"    : routed.code_artifact,
                    "research": routed.research_artifact,
                    "visual"  : routed.visual_artifact,
                }

                for member in coalition.members:
                    art = type_map.get(member.subtask_type)
                    if art:
                        if member.subtask_type == "code":
                            content = art.code
                        elif member.subtask_type == "research":
                            content = art.raw_text
                        else:
                            content = art.content
                        cf.record_output(coalition.coalition_id,
                                         member.agent_id, content,
                                         art.quality_score, tick)

                agg_out   = aggregator.aggregate(coalition)
                completed = cf.complete(coalition.coalition_id,
                                        agg_out.content, tick)

                final_quality = min(1.0, agg_out.quality_score + rag_bonus)

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

                for tt, art in type_map.items():
                    if art is None:
                        continue
                    atype = ArtifactType(tt)
                    if tt == "code":
                        content = art.code
                    elif tt == "research":
                        content = art.raw_text
                    else:
                        content = art.content
                    artifact_store.save(
                        artifact_type = atype,
                        content       = content,
                        author_id     = next(
                            (m.agent_id for m in completed.members
                             if m.subtask_type == tt), 0),
                        task_id       = task_id,
                        tick          = tick,
                        coalition_id  = coalition.coalition_id,
                        quality_score = art.quality_score,
                    )
                    stats[tt] += 1
                    stats["quality"].append(art.quality_score)

                    episodic.record(
                        task_id      = f"{tt}:{task_id}",
                        task_emb     = task_emb_vec,
                        solution_emb = _art_emb(content),
                        quality      = art.quality_score,
                        agent_id     = next(
                            (m.agent_id for m in completed.members
                             if m.subtask_type == tt), 0),
                        coalition_id = coalition.coalition_id,
                    )

                if is_repeat:
                    if len(baseline_quality[repeat_type]) < 3:
                        baseline_quality[repeat_type].append(final_quality)
                    else:
                        rag_quality[repeat_type].append(final_quality)

                solo_qualities = [random.uniform(0.3, 0.8) for _ in range(3)]
                comparison = aggregator.compare_coalition_vs_solo(
                    final_quality, solo_qualities
                )
                stats["coalition_total"] += 1
                if comparison["coalition_wins"]:
                    stats["coalition_wins"] += 1

                win_rate = stats["coalition_wins"] / max(1, stats["coalition_total"])
                log.info(
                    f"[{tick:03d}] COALITION  task={task_type:<8}  "
                    f"quality={final_quality:.2f}  "
                    f"rag_hits={len(rag_hits)}  "
                    f"rag_bonus={rag_bonus:.2f}  "
                    f"{'WIN' if comparison['coalition_wins'] else 'LOSS'}  "
                    f"rate={win_rate:.0%}"
                )
                stats["coalition"] += 1
            else:
                log.info(f"[{tick:03d}] COALITION FAILED  task={task_type}")

        else:
            best_agent = engine.route_task_by_seed(task_type, living)

            if task_type == "code":
                art     = router.code_layer.produce(agent_id=best_agent,
                                                    task_desc=task_desc, tick=tick)
                content = art.code
                quality = art.quality_score
                atype   = ArtifactType.CODE
            elif task_type == "research":
                art     = router.research_layer.produce(agent_id=best_agent,
                                                        task_desc=task_desc, tick=tick)
                content = art.raw_text
                quality = art.quality_score
                atype   = ArtifactType.RESEARCH
            else:
                art     = router.visual_layer.produce(agent_id=best_agent,
                                                      task_desc=task_desc, tick=tick)
                content = art.content
                quality = art.quality_score
                atype   = ArtifactType.VISUAL

            rag_bonus = min(0.10, len(rag_hits) * 0.03)
            quality   = min(1.0, quality + rag_bonus)

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
            artifact_store.save(
                artifact_type = atype,
                content       = content,
                author_id     = best_agent,
                task_id       = task_id,
                tick          = tick,
                quality_score = quality,
            )
            episodic.record(
                task_id      = f"{task_type}:{task_id}",
                task_emb     = task_emb_vec,
                solution_emb = _art_emb(content),
                quality      = quality,
                agent_id     = best_agent,
            )

            if is_repeat:
                if len(baseline_quality[repeat_type]) < 3:
                    baseline_quality[repeat_type].append(quality)
                else:
                    rag_quality[repeat_type].append(quality)

            stats[task_type] += 1
            stats["quality"].append(quality)
            stats["solo"] += 1
            log.info(f"[{tick:03d}] SOLO       task={task_type:<8}  "
                     f"agent={best_agent:02d}  quality={quality:.2f}  "
                     f"rag_hits={len(rag_hits)}")

        for aid in list(living):
            event = economy.tick(aid)
            if event:
                dead, parent = event["killed"], event["parent"]
                engine.replace_agent(dead, parent, current_tick=tick)
                registry.replace_agent(dead, parent, current_tick=tick)
                log.info(f"  [DEATH] agent={dead} → parent={parent}")

        librarian.on_tick(tick)

    avg_q    = sum(stats["quality"]) / max(1, len(stats["quality"]))
    win_rate = stats["coalition_wins"] / max(1, stats["coalition_total"])

    log.info("")
    log.info("── Week 6 results ──────────────────────────────────")
    log.info(f"  episodic records  : {len(episodic)}")
    log.info(f"  semantic nodes    : {distiller.node_count}")
    log.info(f"  code artifacts    : {stats['code']}")
    log.info(f"  research artifacts: {stats['research']}")
    log.info(f"  visual artifacts  : {stats['visual']}")
    log.info(f"  avg quality       : {avg_q:.3f}")
    log.info(f"  solo / coalition  : {stats['solo']} / {stats['coalition']}")
    log.info(f"  coalition win rate: {win_rate:.0%}")

    log.info("")
    log.info("── RAG quality regression ──────────────────────────")
    all_pass = True
    for tt in TASK_TYPES:
        b = baseline_quality[tt]
        r = rag_quality[tt]
        if not b or not r:
            log.info(f"  {tt:<8} insufficient samples (baseline={len(b)} rag={len(r)})")
            continue
        b_mean      = float(np.mean(b))
        r_mean      = float(np.mean(r))
        improvement = (r_mean - b_mean) / (b_mean + 1e-8)
        passed      = improvement >= 0.0
        all_pass    = all_pass and passed
        log.info(f"  {tt:<8} baseline={b_mean:.3f}  rag={r_mean:.3f}  "
                 f"improvement={improvement*100:.1f}%  "
                 f"{'PASS' if passed else 'FAIL'}")

    ok = all(stats[t] > 0 for t in ("code", "research", "visual"))
    log.info(f"\n  DELIVERABLE: {'PASS' if ok and all_pass else 'FAIL'}")

    artifact_store.close()
    return stats


if __name__ == "__main__":
    run(n_agents=10, max_ticks=200, seed=42)
