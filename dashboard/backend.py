import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import random
import uuid
import threading
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

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
from core.society_model   import SocietyModel, SocietyEvent
from core.benchmark       import TASK_BANK

N_AGENTS            = 10
COALITION_THRESHOLD = 0.75
TASK_TYPES          = ["code", "research", "visual"]
TICK_INTERVAL       = 0.1


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
        base = (type_seeds["code"]     if i < 3 else
                type_seeds["research"] if i < 6 else
                type_seeds["visual"])
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


class PantheonSociety:
    def __init__(self, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.type_seeds      = _make_type_seeds(seed=seed)
        self.fingerprints    = _make_fingerprints(self.type_seeds, seed=7)
        self.private_latents = _make_private_latents(seed=99)
        self.living          = list(range(N_AGENTS))
        self.balances        = {i: 100 for i in range(N_AGENTS)}
        self.reputations     = {i: 0.0  for i in range(N_AGENTS)}

        self.engine       = SpeciationEngine(n_agents=N_AGENTS, alpha_scale=0.15,
                                             top_n=3, seed=seed)
        self.evo_engine   = EvolutionEngine(engine=self.engine, registry=None,
                                            evolution_interval=200, mutation_rate=0.10)
        self.cf           = CoalitionFormation()
        self.aggregator   = CoalitionAggregator()
        self.episodic     = EpisodicMemory()
        self.distiller    = KnowledgeDistiller(episodic=self.episodic)
        self.librarian    = LibrarianAgent(episodic=self.episodic,
                                           distiller=self.distiller)
        self.society_model= SocietyModel(d_model=64, n_heads=4, n_layers=2, ctx=32)
        self.historian    = HistorianAgent(
            episodic=self.episodic, distiller=self.distiller,
            engine=self.engine, society_model=self.society_model,
            report_interval=100,
        )
        self.historian.link_balances(self.balances)
        self.router       = OutputRouter(visual_mode="ascii", episodic=self.episodic)

        self.task_pool    = list(TASK_BANK) * 100
        random.shuffle(self.task_pool)

        self.tick              = 0
        self.active_coalitions = []
        self.recent_artifacts  = []
        self.event_log         = []
        self.roles             = {}
        self.running           = False
        self._lock             = threading.Lock()

    def step(self):
        self.tick += 1
        tick = self.tick
        task_type, task_desc, difficulty, reward = self.task_pool[tick % len(self.task_pool)]
        task_id  = str(uuid.uuid4())
        task_emb = _embed(tick)

        self.active_coalitions = []
        quality = 0.0
        mode    = "SOLO"

        if difficulty >= COALITION_THRESHOLD:
            decomp  = TaskDecomposer(min_types=3).decompose(
                task_desc, task_id, difficulty=difficulty
            )
            ae      = AuctionEngine()
            results = ae.run_all_auctions(
                subtasks=decomp.subtasks, living_agents=self.living,
                fingerprints=self.fingerprints, token_balances=self.balances,
                type_seeds=self.type_seeds,
            )
            for r in results:
                if r.has_winner:
                    self.balances[r.winner_id] = max(
                        0, self.balances[r.winner_id] - r.tokens_spent
                    )

            coalition = self.cf.form(
                parent_task_id=task_id, auction_results=results,
                subtasks=decomp.subtasks, reputations=self.reputations,
                private_latents=self.private_latents,
                token_balances=self.balances, tick=tick,
            )

            if coalition is not None:
                self.active_coalitions = [
                    {"coalition_id": coalition.coalition_id,
                     "members"     : coalition.member_ids,
                     "coordinator" : coalition.coordinator_id,
                     "task_type"   : task_type}
                ]
                routed = self.router.route_and_produce(
                    coalition=coalition, task_desc=task_desc, tick=tick,
                    fingerprints=self.fingerprints, type_seeds=self.type_seeds,
                    difficulty=difficulty, task_emb=task_emb,
                )
                quality = routed.mean_quality
                mode    = "COALITION"

                type_map = {
                    "code"    : routed.code_artifact,
                    "research": routed.research_artifact,
                    "visual"  : routed.visual_artifact,
                }
                for member in coalition.members:
                    art = type_map.get(member.subtask_type)
                    if art:
                        content = (art.code if member.subtask_type == "code"
                                   else art.raw_text if member.subtask_type == "research"
                                   else art.content)
                        self.cf.record_output(coalition.coalition_id,
                                              member.agent_id, content,
                                              art.quality_score, tick)

                agg_out   = self.aggregator.aggregate(coalition)
                completed = self.cf.complete(coalition.coalition_id, agg_out.content, tick)

                for member in completed.members:
                    if member.reward_share > 0:
                        self.balances[member.agent_id] += member.reward_share
                    self.reputations[member.agent_id] = (
                        0.9 * self.reputations[member.agent_id]
                        + 0.1 * member.quality_score
                    )
                    self.engine.update_fingerprint(
                        agent_id=member.agent_id, task_type=member.subtask_type,
                        success=member.quality_score >= 0.5,
                        reward=member.reward_share, tick=tick,
                    )

                self.episodic.record(
                    task_id=task_id, task_emb=task_emb,
                    solution_emb=_embed(hash(agg_out.content) % (2**31)),
                    quality=quality,
                    agent_id=completed.members[0].agent_id if completed.members else 0,
                    coalition_id=coalition.coalition_id,
                )

                self.recent_artifacts.append({
                    "tick": tick, "type": task_type,
                    "quality": round(quality, 3), "mode": mode,
                    "coalition_id": coalition.coalition_id[:8],
                })
                if len(self.recent_artifacts) > 20:
                    self.recent_artifacts = self.recent_artifacts[-20:]

                ev = SocietyEvent(tick=tick, event_type="COALITION_FORMED",
                                  agent_id=coalition.coordinator_id,
                                  quality=quality, success=quality >= 0.5)
                self.society_model.observe(ev)
                self.balances = self.society_model.curiosity_check(ev, self.balances)
                self.historian.log_task(quality, is_coalition=True)

        else:
            best_agent = self.engine.route_task_by_seed(task_type, self.living)
            if task_type == "code":
                art     = self.router.code_layer.produce(
                    agent_id=best_agent, task_desc=task_desc,
                    tick=tick, difficulty=difficulty)
                quality = art.quality_score
                content = art.code
            elif task_type == "research":
                art     = self.router.research_layer.produce(
                    agent_id=best_agent, task_desc=task_desc,
                    tick=tick, difficulty=difficulty)
                quality = art.quality_score
                content = art.raw_text
            else:
                art     = self.router.visual_layer.produce(
                    agent_id=best_agent, task_desc=task_desc,
                    tick=tick, difficulty=difficulty)
                quality = art.quality_score
                content = art.content

            if quality >= 0.5:
                self.balances[best_agent] += reward
            else:
                self.balances[best_agent] = max(
                    0, self.balances[best_agent] - max(1, int(difficulty * 7))
                )
            self.reputations[best_agent] = (
                0.9 * self.reputations[best_agent] + 0.1 * quality
            )
            self.engine.update_fingerprint(
                agent_id=best_agent, task_type=task_type,
                success=quality >= 0.5,
                reward=reward if quality >= 0.5 else 0, tick=tick,
            )
            self.episodic.record(
                task_id=task_id, task_emb=task_emb,
                solution_emb=_embed(hash(content) % (2**31)),
                quality=quality, agent_id=best_agent,
            )
            self.recent_artifacts.append({
                "tick": tick, "type": task_type,
                "quality": round(quality, 3), "mode": mode,
                "agent_id": best_agent,
            })
            if len(self.recent_artifacts) > 20:
                self.recent_artifacts = self.recent_artifacts[-20:]

            ev = SocietyEvent(
                tick=tick,
                event_type="TASK_SUCCESS" if quality >= 0.5 else "TASK_FAIL",
                agent_id=best_agent, quality=quality, success=quality >= 0.5,
            )
            self.society_model.observe(ev)
            self.balances = self.society_model.curiosity_check(ev, self.balances)
            self.historian.log_task(quality, is_coalition=False)

        evo_report = self.evo_engine.maybe_evolve(tick, self.balances, self.living)
        if evo_report and evo_report.n_replaced > 0:
            self.historian.log_evolution()

        self.historian.on_tick(tick)
        self.librarian.on_tick(tick)

        roles = {}
        for aid, rec in self.engine.records.items():
            fp   = rec.fingerprint
            best = max(self.engine._type_seeds.items(),
                       key=lambda kv: float(fp @ kv[1]))
            roles.setdefault(best[0], []).append(aid)
        self.roles = roles

    def snapshot(self):
        agents = []
        for aid in self.living:
            rec  = self.engine.records.get(aid)
            role = next((r for r, ids in self.roles.items() if aid in ids), "unknown")
            agents.append({
                "id"        : aid,
                "tokens"    : self.balances.get(aid, 0),
                "reputation": round(self.reputations.get(aid, 0.0), 3),
                "role"      : role,
            })
        return {
            "tick"              : self.tick,
            "agents"            : agents,
            "active_coalitions" : self.active_coalitions,
            "economy"           : self.balances,
            "recent_artifacts"  : self.recent_artifacts[-5:],
            "episodic_count"    : len(self.episodic),
            "semantic_nodes"    : self.distiller.node_count,
            "society_model"     : self.society_model.snapshot(),
            "roles"             : {k: len(v) for k, v in self.roles.items()},
        }


society = PantheonSociety(seed=42)
app     = FastAPI()


@app.get("/")
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    return FileResponse(html_path)


@app.get("/snapshot")
async def get_snapshot():
    return society.snapshot()


@app.websocket("/ws/society")
async def society_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            society.step()
            snap = society.snapshot()
            await ws.send_text(json.dumps(snap))
            await asyncio.sleep(TICK_INTERVAL)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run("dashboard.backend:app", host="0.0.0.0", port=8000, reload=False)

