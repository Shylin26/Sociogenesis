import sys
import os
import time
import webbrowser
import threading
import subprocess
import uuid
import random
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

DEMO_SEED    = 42
N_AGENTS     = 20
WARMUP_TICKS = 500
DEMO_TIMEOUT = 60
TASK_TYPES   = ["code", "research", "visual"]
COALITION_THRESHOLD = 0.75

HARD_PROBLEM = (
    "Build a Python web scraper for Hacker News front page. "
    "Write a hypothesis about what topics dominate today. "
    "Generate a data flow diagram of the scraper architecture."
)


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


def _print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           SOCIOGENESIS — Personal AGI Civilization       ║")
    print("║                                                          ║")
    print("║  20 identical agents. One hard problem. No assigned      ║")
    print("║  roles. Watch them specialise, form coalitions, and      ║")
    print("║  solve it together. Runs entirely on this MacBook.       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def _print_section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def build_society():
    random.seed(DEMO_SEED)
    torch.manual_seed(DEMO_SEED)
    np.random.seed(DEMO_SEED)

    type_seeds      = _make_type_seeds(seed=DEMO_SEED)
    fingerprints    = _make_fingerprints(type_seeds)
    private_latents = _make_private_latents()
    living          = list(range(N_AGENTS))
    balances        = {i: 100 for i in range(N_AGENTS)}
    reputations     = {i: 0.0  for i in range(N_AGENTS)}

    engine        = SpeciationEngine(n_agents=N_AGENTS, alpha_scale=0.25,
                                     top_n=3, seed=DEMO_SEED)
    evo_engine    = EvolutionEngine(engine=engine, registry=None,
                                    evolution_interval=100, mutation_rate=0.10)
    cf            = CoalitionFormation()
    aggregator    = CoalitionAggregator()
    episodic      = EpisodicMemory()
    distiller     = KnowledgeDistiller(episodic=episodic)
    librarian     = LibrarianAgent(episodic=episodic, distiller=distiller)
    society_model = SocietyModel(d_model=64, n_heads=4, n_layers=2, ctx=32)
    historian     = HistorianAgent(
        episodic=episodic, distiller=distiller,
        engine=engine, society_model=society_model,
        report_interval=100,
    )
    historian.link_balances(balances)
    router = OutputRouter(visual_mode="ascii", episodic=episodic)

    return dict(
        type_seeds=type_seeds, fingerprints=fingerprints,
        private_latents=private_latents, living=living,
        balances=balances, reputations=reputations,
        engine=engine, evo_engine=evo_engine, cf=cf,
        aggregator=aggregator, episodic=episodic, distiller=distiller,
        librarian=librarian, society_model=society_model,
        historian=historian, router=router,
    )


def run_tick(s, tick, task_type, task_desc, difficulty, reward):
    task_id  = str(uuid.uuid4())
    task_emb = _embed(tick)

    if difficulty >= COALITION_THRESHOLD:
        decomp  = TaskDecomposer(min_types=3).decompose(task_desc, task_id, difficulty=difficulty)
        ae      = AuctionEngine()
        results = ae.run_all_auctions(
            subtasks=decomp.subtasks, living_agents=s['living'],
            fingerprints=s['fingerprints'], token_balances=s['balances'],
            type_seeds=s['type_seeds'],
        )
        for r in results:
            if r.has_winner:
                s['balances'][r.winner_id] = max(0, s['balances'][r.winner_id] - r.tokens_spent)

        coalition = s['cf'].form(
            parent_task_id=task_id, auction_results=results,
            subtasks=decomp.subtasks, reputations=s['reputations'],
            private_latents=s['private_latents'],
            token_balances=s['balances'], tick=tick,
        )

        if coalition is not None:
            routed = s['router'].route_and_produce(
                coalition=coalition, task_desc=task_desc, tick=tick,
                fingerprints=s['fingerprints'], type_seeds=s['type_seeds'],
                difficulty=difficulty, task_emb=task_emb,
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
                    content = (art.code if member.subtask_type=="code"
                               else art.raw_text if member.subtask_type=="research"
                               else art.content)
                    s['cf'].record_output(coalition.coalition_id,
                                          member.agent_id, content,
                                          art.quality_score, tick)

            agg_out   = s['aggregator'].aggregate(coalition)
            completed = s['cf'].complete(coalition.coalition_id, agg_out.content, tick)

            for member in completed.members:
                if member.reward_share > 0:
                    s['balances'][member.agent_id] += member.reward_share
                s['reputations'][member.agent_id] = (
                    0.9 * s['reputations'][member.agent_id] + 0.1 * member.quality_score
                )
                s['engine'].update_fingerprint(
                    agent_id=member.agent_id, task_type=member.subtask_type,
                    success=member.quality_score >= 0.5,
                    reward=member.reward_share, tick=tick,
                )

            s['episodic'].record(
                task_id=task_id, task_emb=task_emb,
                solution_emb=_embed(hash(agg_out.content) % (2**31)),
                quality=quality,
                agent_id=completed.members[0].agent_id if completed.members else 0,
                coalition_id=coalition.coalition_id,
            )
            s['historian'].log_task(quality, is_coalition=True)

            ev = SocietyEvent(tick=tick, event_type="COALITION_FORMED",
                              agent_id=coalition.coordinator_id,
                              quality=quality, success=quality>=0.5)
            s['society_model'].observe(ev)
            s['balances'] = s['society_model'].curiosity_check(ev, s['balances'])
            return coalition, routed, quality

    else:
        best_agent = s['engine'].route_task_by_seed(task_type, s['living'])
        if task_type == "code":
            art = s['router'].code_layer.produce(agent_id=best_agent,
                  task_desc=task_desc, tick=tick, difficulty=difficulty)
            quality, content = art.quality_score, art.code
        elif task_type == "research":
            art = s['router'].research_layer.produce(agent_id=best_agent,
                  task_desc=task_desc, tick=tick, difficulty=difficulty)
            quality, content = art.quality_score, art.raw_text
        else:
            art = s['router'].visual_layer.produce(agent_id=best_agent,
                  task_desc=task_desc, tick=tick, difficulty=difficulty)
            quality, content = art.quality_score, art.content

        if quality >= 0.5:
            s['balances'][best_agent] += reward
        else:
            s['balances'][best_agent] = max(0, s['balances'][best_agent] - max(1, int(difficulty*7)))

        s['reputations'][best_agent] = 0.9*s['reputations'][best_agent] + 0.1*quality
        s['engine'].update_fingerprint(
            agent_id=best_agent, task_type=task_type,
            success=quality>=0.5, reward=reward if quality>=0.5 else 0, tick=tick,
        )
        s['episodic'].record(
            task_id=task_id, task_emb=task_emb,
            solution_emb=_embed(hash(content) % (2**31)),
            quality=quality, agent_id=best_agent,
        )
        s['historian'].log_task(quality, is_coalition=False)

        ev = SocietyEvent(tick=tick,
                          event_type="TASK_SUCCESS" if quality>=0.5 else "TASK_FAIL",
                          agent_id=best_agent, quality=quality, success=quality>=0.5)
        s['society_model'].observe(ev)
        s['balances'] = s['society_model'].curiosity_check(ev, s['balances'])
        return None, None, quality

    evo = s['evo_engine'].maybe_evolve(tick, s['balances'], s['living'])
    if evo and evo.n_replaced > 0:
        s['historian'].log_evolution()
    s['historian'].on_tick(tick)
    s['librarian'].on_tick(tick)


def warmup(s):
    _print_section(f"WARMUP — {WARMUP_TICKS} ticks to build specialist roles")
    task_pool = list(TASK_BANK) * (WARMUP_TICKS // len(TASK_BANK) + 1)
    random.shuffle(task_pool)

    for tick in range(1, WARMUP_TICKS + 1):
        task_type, task_desc, difficulty, reward = task_pool[tick-1]
        run_tick(s, tick, task_type, task_desc, difficulty, reward)

        evo = s['evo_engine'].maybe_evolve(tick, s['balances'], s['living'])
        if evo and evo.n_replaced > 0:
            s['historian'].log_evolution()
            for e in evo.events:
                print(f"  [EVOLUTION] Agent {e.dead_id} → parent Agent {e.parent_id}")

        s['historian'].on_tick(tick)
        s['librarian'].on_tick(tick)

        if tick % 50 == 0:
            roles = {}
            for aid, rec in s['engine'].records.items():
                fp   = rec.fingerprint
                best = max(s['engine']._type_seeds.items(),
                           key=lambda kv: float(fp @ kv[1]))
                roles.setdefault(best[0], []).append(aid)
            print(f"  tick {tick:4d} | roles: " +
                  " · ".join(f"{r}: {len(v)} agents" for r,v in roles.items()))

    print(f"\n  Warmup complete. {len(s['episodic'])} memories stored.")


def solve_hard_problem(s):
    _print_section("HARD PROBLEM")
    print(f"\n  {HARD_PROBLEM}\n")
    print("  This requires: code agent + research agent + visual agent")
    print("  in a coalition, with a coordinator merging outputs.\n")

    start   = time.time()
    tick    = WARMUP_TICKS + 1
    result  = None

    while time.time() - start < DEMO_TIMEOUT:
        coalition, routed, quality = run_tick(
            s, tick, "code", HARD_PROBLEM, 0.90, 25
        )
        if coalition is not None and routed is not None:
            result = (coalition, routed, quality)
            break
        tick += 1
        time.sleep(0.05)

    return result, tick


def print_result(result, tick, s):
    if result is None:
        print("\n  Coalition did not form in time.")
        return

    coalition, routed, quality = result
    _print_section("SOCIOGENESIS RESULT")

    print(f"\n  Coalition formed:  {coalition.member_ids}")
    print(f"  Coordinator:       Agent {coalition.coordinator_id}")
    print(f"  Ticks elapsed:     {tick - WARMUP_TICKS}")
    print(f"  Mean quality:      {quality:.3f}")
    print()

    if routed.code_artifact:
        print(f"  CODE ARTIFACT      (quality={routed.code_artifact.quality_score:.3f})")
        print(f"  {'─'*50}")
        lines = routed.code_artifact.code.strip().split('\n')[:8]
        for l in lines: print(f"  {l}")
        if len(routed.code_artifact.code.strip().split('\n')) > 8:
            print(f"  ... ({len(routed.code_artifact.code.strip().split(chr(10)))} lines total)")
        print()

    if routed.research_artifact:
        print(f"  RESEARCH ARTIFACT  (quality={routed.research_artifact.quality_score:.3f})")
        print(f"  {'─'*50}")
        print(f"  {routed.research_artifact.raw_text.strip()[:300]}")
        print()

    if routed.visual_artifact:
        print(f"  VISUAL ARTIFACT    (quality={routed.visual_artifact.quality_score:.3f})")
        print(f"  {'─'*50}")
        print(f"  {routed.visual_artifact.content.strip()[:300]}")
        print()

    top3 = sorted(s['balances'].items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  TOP EARNERS:  " + "  ".join(f"A{a}={t} tokens" for a,t in top3))
    print(f"  MEMORIES:     {len(s['episodic'])} episodic records")
    print(f"  SEMANTIC:     {s['distiller'].node_count} knowledge nodes")
    print()


def generate_paper(s):
    _print_section("GENERATING SOCIETY PAPER")
    writer = PaperWriter(
        historian_reports=s['historian'].all_reports(),
        society_snapshot=s['historian'].snapshot(),
    )
    paper    = writer.build_paper(s['living'])
    json_path= writer.save_json(paper, "artifacts/demo_paper.json")
    pdf_path = writer.save_pdf(paper,  "artifacts/demo_paper.pdf")
    print(f"\n  Paper generated: {pdf_path}")
    print(f"  Word count:      {paper.word_count()}")
    print(f"  Sections:        {list(paper.sections.keys())}")
    print(f"  Debate:          {len(paper.debate)} agents argued about specialisation")
    return pdf_path


def launch_dashboard():
    _print_section("LAUNCHING DASHBOARD")
    print("\n  Starting dashboard at http://localhost:8000 ...")

    def _run():
        env = os.environ.copy()
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        subprocess.Popen(
            [sys.executable, '-m', 'dashboard.backend'],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(3)
    webbrowser.open('http://localhost:8000')
    print("  Dashboard open. Press Ctrl+C to stop.\n")


def run_demo():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    _print_banner()

    print("  Building society with 20 agents (seed=42)...")
    s = build_society()
    print("  Done.\n")

    warmup(s)
    result, tick = solve_hard_problem(s)
    print_result(result, tick, s)
    generate_paper(s)
    launch_dashboard()

    print("=" * 62)
    print("  SOCIOGENESIS — one sentence:")
    print()
    print("  We gave 20 identical agents a hard problem and a shared")
    print("  economy. They organised themselves into specialists,")
    print("  formed coalitions, produced code, research, and visuals")
    print("  — and wrote a paper about themselves.")
    print("  It runs on a MacBook.")
    print("=" * 62)
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    run_demo()