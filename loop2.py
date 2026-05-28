import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import uuid
import logging
import torch
import torch.nn.functional as F
import numpy as np

from speciation.engine    import SpeciationEngine
from speciation.evolution import EvolutionEngine
from coalition.task_decomposer import TaskDecomposer
from coalition.Auction    import AuctionEngine
from coalition.coalition  import CoalitionFormation
from coalition.aggregator import CoalitionAggregator
from output.output_router import OutputRouter
from memory.episodic      import EpisodicMemory, DIM
from memory.distillation  import KnowledgeDistiller
from memory.librarian     import LibrarianAgent
from memory.historian     import HistorianAgent
from memory.paper_writer  import PaperWriter
from core.society_model   import SocietyModel, SocietyEvent
from core.benchmark       import TASK_BANK

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

N_AGENTS            = 10
TICKS               = 1000
COALITION_THRESHOLD = 0.75
TASK_TYPES          = ["code", "research", "visual"]


def _make_type_seeds(seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return {
        name: F.normalize(torch.randn(128, generator=gen), dim=0)
        for name in TASK_TYPES
    }


def _make_fingerprints(type_seeds, n_agents=N_AGENTS, seed=7):
    gen = torch.Generator()
    gen.manual_seed(seed)
    fps = {}
    for i in range(n_agents):
        base = F.normalize(torch.randn(128, generator=gen), dim=0)
        fps[i] = F.normalize(base + torch.randn(128, generator=gen) * 0.1, dim=0)
    return fps


def _make_private_latents(n_agents=N_AGENTS, seed=99):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return {
        i: F.normalize(torch.randn(128, generator=gen), dim=0)
        for i in range(n_agents)
    }


def _embed(seed_int):
    rng = np.random.RandomState(seed_int % (2 ** 31))
    v   = rng.randn(DIM).astype(np.float32)
    v  /= np.linalg.norm(v) + 1e-8
    return v


def run(n_agents=N_AGENTS, max_ticks=TICKS, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    type_seeds      = _make_type_seeds(seed=seed)
    fingerprints    = _make_fingerprints(type_seeds, seed=7)
    private_latents = _make_private_latents(seed=99)
    living          = list(range(n_agents))
    balances        = {i: 100 for i in range(n_agents)}
    reputations     = {i: 0.0  for i in range(n_agents)}

    engine        = SpeciationEngine(n_agents=n_agents, alpha_scale=0.15,
                                     top_n=3, seed=seed)
    evo_engine    = EvolutionEngine(
                        engine            = engine,
                        registry          = None,
                        evolution_interval= 200,
                        mutation_rate     = 0.10,
                    )
    cf            = CoalitionFormation()
    aggregator    = CoalitionAggregator()
    episodic      = EpisodicMemory()
    distiller     = KnowledgeDistiller(episodic=episodic)
    librarian     = LibrarianAgent(episodic=episodic, distiller=distiller)
    society_model = SocietyModel(d_model=64, n_heads=4, n_layers=2, ctx=32)
    historian     = HistorianAgent(
                        episodic       = episodic,
                        distiller      = distiller,
                        engine         = engine,
                        society_model  = society_model,
                        report_interval= 100,
                    )
    historian.link_balances(balances)
    router        = OutputRouter(visual_mode="ascii", episodic=episodic)

    task_pool     = list(TASK_BANK) * (max_ticks // len(TASK_BANK) + 1)
    random.shuffle(task_pool)

    stats = {
        "quality"         : [],
        "coalition_wins"  : 0,
        "coalition_total" : 0,
        "solo"            : 0,
        "evolution_events": 0,
    }

    log.info("=" * 55)
    log.info("PANTHEON  Week 8 — Historian + Meta-Output")
    log.info("=" * 55)

    for tick in range(1, max_ticks + 1):
        task_type, task_desc, difficulty, reward = task_pool[tick - 1]
        task_id  = str(uuid.uuid4())
        task_emb = _embed(tick)

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
                    task_emb     = task_emb,
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
                        if member.subtask_type == "code":
                            output = art.code
                        elif member.subtask_type == "research":
                            output = art.raw_text
                        else:
                            output = art.content
                        cf.record_output(
                            coalition.coalition_id,
                            member.agent_id,
                            output,
                            art.quality_score,
                            tick,
                        )

                agg_out   = aggregator.aggregate(coalition)
                completed = cf.complete(coalition.coalition_id,
                                        agg_out.content, tick)

                for member in completed.members:
                    if member.reward_share > 0:
                        balances[member.agent_id] += member.reward_share
                    reputations[member.agent_id] = (
                        0.9 * reputations[member.agent_id]
                        + 0.1 * member.quality_score
                    )
                    engine.update_fingerprint(
                        agent_id  = member.agent_id,
                        task_type = member.subtask_type,
                        success   = member.quality_score >= 0.5,
                        reward    = member.reward_share,
                        tick      = tick,
                    )

                episodic.record(
                    task_id      = task_id,
                    task_emb     = task_emb,
                    solution_emb = _embed(hash(agg_out.content) % (2 ** 31)),
                    quality      = quality,
                    agent_id     = completed.members[0].agent_id
                                   if completed.members else 0,
                    coalition_id = coalition.coalition_id,
                )

                solo_qualities = [random.uniform(0.3, 0.8) for _ in range(3)]
                comparison     = aggregator.compare_coalition_vs_solo(
                    agg_out.quality_score, solo_qualities
                )
                stats["coalition_total"] += 1
                if comparison["coalition_wins"]:
                    stats["coalition_wins"] += 1

                historian.log_task(quality, is_coalition=True)
                stats["quality"].append(quality)

                ev = SocietyEvent(
                    tick       = tick,
                    event_type = "COALITION_FORMED",
                    agent_id   = coalition.coordinator_id,
                    quality    = quality,
                    success    = quality >= 0.5,
                )
                society_model.observe(ev)
                balances = society_model.curiosity_check(ev, balances)

                win_rate = stats["coalition_wins"] / max(1, stats["coalition_total"])
                log.info(
                    f"[{tick:04d}] COALITION  task={task_type:<8}  "
                    f"quality={quality:.2f}  "
                    f"{'WIN' if comparison['coalition_wins'] else 'LOSS'}  "
                    f"rate={win_rate:.0%}"
                )

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
            engine.update_fingerprint(
                agent_id  = best_agent,
                task_type = task_type,
                success   = quality >= 0.5,
                reward    = reward if quality >= 0.5 else 0,
                tick      = tick,
            )

            episodic.record(
                task_id      = task_id,
                task_emb     = task_emb,
                solution_emb = _embed(hash(content) % (2 ** 31)),
                quality      = quality,
                agent_id     = best_agent,
            )

            historian.log_task(quality, is_coalition=False)
            stats["quality"].append(quality)
            stats["solo"] += 1

            ev = SocietyEvent(
                tick       = tick,
                event_type = "TASK_SUCCESS" if quality >= 0.5 else "TASK_FAIL",
                agent_id   = best_agent,
                quality    = quality,
                success    = quality >= 0.5,
            )
            society_model.observe(ev)
            balances = society_model.curiosity_check(ev, balances)

            log.info(
                f"[{tick:04d}] SOLO       task={task_type:<8}  "
                f"agent={best_agent:02d}  quality={quality:.2f}"
            )

        evo_report = evo_engine.maybe_evolve(tick, balances, living)
        if evo_report and evo_report.n_replaced > 0:
            stats["evolution_events"] += evo_report.n_replaced
            historian.log_evolution()
            for ev_event in evo_report.events:
                ev = SocietyEvent(
                    tick=tick, event_type="AGENT_DIED",
                    agent_id=ev_event.dead_id, quality=0.0, success=False,
                )
                society_model.observe(ev)
                ev2 = SocietyEvent(
                    tick=tick, event_type="AGENT_BORN",
                    agent_id=ev_event.dead_id, quality=0.0, success=True,
                )
                society_model.observe(ev2)
            log.info(
                f"  [EVOLUTION] tick={tick}  "
                f"replaced={evo_report.n_replaced}  "
                f"events={[(e.dead_id, e.parent_id) for e in evo_report.events]}"
            )

        historian_report = historian.on_tick(tick)
        if historian_report:
            log.info(historian_report.summary())

        librarian.on_tick(tick)

    avg_q    = sum(stats["quality"]) / max(1, len(stats["quality"]))
    win_rate = stats["coalition_wins"] / max(1, stats["coalition_total"])

    log.info("")
    log.info("── Generating society paper ────────────────────────")

    writer = PaperWriter(
        historian_reports = historian.all_reports(),
        society_snapshot  = historian.snapshot(),
    )
    paper    = writer.build_paper(living)
    json_path= writer.save_json(paper, "artifacts/pantheon_paper.json")
    pdf_path = writer.save_pdf(paper,  "artifacts/pantheon_paper.pdf")

    log.info(f"  paper sections : {list(paper.sections.keys())}")
    log.info(f"  word count     : {paper.word_count()}")
    log.info(f"  debate args    : {len(paper.debate)}")
    log.info(f"  figures        : {len(paper.figures)}")
    log.info(f"  JSON           : {json_path}")
    log.info(f"  PDF            : {pdf_path}")

    log.info("")
    log.info("── Week 8 results ──────────────────────────────────")
    log.info(f"  avg quality       : {avg_q:.3f}")
    log.info(f"  coalition win rate: {win_rate:.0%}")
    log.info(f"  evolution events  : {stats['evolution_events']}")
    log.info(f"  historian reports : {historian.snapshot()}")
    log.info(f"  paper word count  : {paper.word_count()}")

    passed = (
        avg_q >= 0.75
        and win_rate >= 0.60
        and paper.word_count() >= 300
        and len(paper.debate) == 3
        and os.path.exists(pdf_path)
        and os.path.exists(json_path)
    )
    log.info(f"\n  DELIVERABLE: {'PASS' if passed else 'FAIL'}")
    return stats, paper


if __name__ == "__main__":
    run(n_agents=N_AGENTS, max_ticks=TICKS, seed=42)