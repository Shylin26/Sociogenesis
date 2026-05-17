import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


class TaskEncoder(nn.Module):
    INPUT_DIM  = 256
    HIDDEN_DIM = 256
    OUTPUT_DIM = 128

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def encode_string(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.INPUT_DIM)
        for i, ch in enumerate(text[:512]):
            bucket = ord(ch) % self.INPUT_DIM
            vec[bucket] += 1.0 / (i + 1)
        norm = vec.norm()
        return vec / norm if norm > 0 else vec

    def forward(self, text: str) -> torch.Tensor:
        x   = self.encode_string(text).unsqueeze(0)
        out = self.net(x).squeeze(0)
        return F.normalize(out, dim=0)


@dataclass
class AgentTaskRecord:
    agent_id     : int
    fingerprint  : torch.Tensor
    successes    : dict = field(default_factory=dict)
    update_count : int  = 0
    HISTORY      : int  = 30


class SpeciationEngine:
    TASK_TYPE_IDX = {"code": 0, "research": 1, "visual": 2}

    def __init__(self, n_agents    : int,
                 fp_dim      : int   = 128,
                 alpha_scale : float = 0.05,
                 top_n       : int   = 3,
                 seed        : int   = 42):

        self.fp_dim      = fp_dim
        self.alpha_scale = alpha_scale
        self.top_n       = top_n

        gen = torch.Generator()
        gen.manual_seed(seed)
        self._type_seeds: dict[str, torch.Tensor] = {}
        for name in self.TASK_TYPE_IDX:
            raw = torch.randn(fp_dim, generator=gen)
            self._type_seeds[name] = F.normalize(raw, dim=0)

        names = list(self._type_seeds.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                sim = torch.dot(self._type_seeds[names[i]],
                                self._type_seeds[names[j]]).item()
                assert abs(sim) < 0.5, \
                    f"Seeds {names[i]}/{names[j]} too similar: {sim:.3f}"

        self.records: dict[int, AgentTaskRecord] = {}
        for i in range(n_agents):
            init_type = names[i % len(names)]
            seed_fp   = self._type_seeds[init_type]
            noise     = torch.randn(fp_dim, generator=gen) * 0.3
            fp        = F.normalize(seed_fp + noise, dim=0)
            self.records[i] = AgentTaskRecord(agent_id=i, fingerprint=fp)

        self.encoder = TaskEncoder()
        self._attractors: dict[str, torch.Tensor] = dict(self._type_seeds)

        self.fingerprint_log: list[dict] = []
        self.total_updates = 0
        self.total_routes  = 0
        self._task_counts: dict[str, int] = {}

    def update_fingerprint(self, agent_id  : int,
                           task_type : str,
                           success   : bool,
                           reward    : int,
                           tick      : int) -> Optional[torch.Tensor]:
        rec = self.records[agent_id]
        self._task_counts[task_type] = self._task_counts.get(task_type, 0) + 1

        if success:
            if task_type not in rec.successes:
                rec.successes[task_type] = []
            rec.successes[task_type].append(reward)
            if len(rec.successes[task_type]) > rec.HISTORY:
                rec.successes[task_type].pop(0)

            self._recompute_attractor(task_type)

            attractor = self._attractors[task_type]
            alpha     = max(0.01, min(0.4, self.alpha_scale * reward))

            fp   = rec.fingerprint
            fp   = fp + alpha * (attractor - fp)
            norm = fp.norm()
            fp   = fp / norm if norm > 1e-8 else \
                   F.normalize(torch.randn(self.fp_dim), dim=0)

            rec.fingerprint  = fp
            rec.update_count += 1
            self.total_updates += 1
            return fp

        return None

    def _recompute_attractor(self, task_type: str):
        candidates = []
        for agent_id, rec in self.records.items():
            count = len(rec.successes.get(task_type, []))
            if count > 0:
                candidates.append((agent_id, count))

        seed = self._type_seeds[task_type]

        if not candidates:
            self._attractors[task_type] = seed
            return

        candidates.sort(key=lambda x: x[1], reverse=True)
        top  = candidates[:self.top_n]
        fps  = [self.records[aid].fingerprint for aid, _ in top]
        mean = torch.stack(fps).mean(dim=0)
        norm = mean.norm()
        if norm > 1e-8:
            mean = mean / norm

        total = sum(len(self.records[aid].successes.get(task_type, []))
                    for aid in self.records)
        seed_w = max(0.2, 0.9 - 0.7 * (total / 200.0))

        blended = seed_w * seed + (1 - seed_w) * mean
        norm    = blended.norm()
        if norm > 1e-8:
            self._attractors[task_type] = blended / norm

    def route_task_by_seed(self, task_type: str,
                           living_agents: list[int]) -> int:
        """Route using the fixed type seed directly.
        More reliable than encoder before fingerprints have converged.
        After tick ~200, fingerprints and seeds align anyway.
        """
        if not living_agents:
            return 0
        seed = self._type_seeds.get(task_type)
        if seed is None:
            return living_agents[0]
        best_id, best_score = living_agents[0], -999.0
        for aid in living_agents:
            fp    = self.records[aid].fingerprint
            score = torch.dot(seed, fp).item()
            if score > best_score:
                best_score, best_id = score, aid
        self.total_routes += 1
        return best_id

    def route_task(self, task_description: str,
                   task_type       : str,
                   living_agents   : list[int]) -> int:
        if not living_agents:
            return 0

        with torch.no_grad():
            task_emb = self.encoder(task_description)

        best_id, best_score = living_agents[0], -999.0
        for aid in living_agents:
            score = torch.dot(task_emb, self.records[aid].fingerprint).item()
            if score > best_score:
                best_score, best_id = score, aid

        self.total_routes += 1
        return best_id

    def route_top_k(self, task_description: str,
                    task_type       : str,
                    living_agents   : list[int],
                    k               : int = 3) -> list[int]:
        if not living_agents:
            return []
        with torch.no_grad():
            task_emb = self.encoder(task_description)
        scored = [(aid, torch.dot(task_emb,
                   self.records[aid].fingerprint).item())
                  for aid in living_agents]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored[:k]]

    def sync_to_registry(self, registry) -> None:
        for agent_id, rec in self.records.items():
            registry.records[agent_id].skill_fingerprint = \
                rec.fingerprint.tolist()

    def replace_agent(self, dead_id: int, parent_id: int,
                      current_tick: int, mutation_rate: float = 0.1):
        parent = self.records[parent_id]
        new_fp = parent.fingerprint + mutation_rate * torch.randn(self.fp_dim)
        new_fp = F.normalize(new_fp, dim=0)
        self.records[dead_id] = AgentTaskRecord(
            agent_id    = dead_id,
            fingerprint = new_fp,
            successes   = {k: v[len(v)//2:]
                           for k, v in parent.successes.items()},
        )

    def all_fingerprints(self) -> torch.Tensor:
        fps = [self.records[i].fingerprint
               for i in sorted(self.records.keys())]
        return torch.stack(fps)

    def snapshot_fingerprints(self, tick: int):
        self.fingerprint_log.append({
            "tick"        : tick,
            "fingerprints": self.all_fingerprints().tolist(),
            "agent_ids"   : sorted(self.records.keys()),
        })

    def cluster_summary(self) -> dict:
        fps   = self.all_fingerprints()
        n     = fps.shape[0]
        sims  = []
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                sim = torch.dot(fps[i], fps[j]).item()
                sims.append(sim)
                pairs.append((i, j, sim))
        if not sims:
            return {}
        pairs.sort(key=lambda x: x[2])
        return {
            "mean_pairwise_sim"   : round(sum(sims)/len(sims), 4),
            "min_sim"             : round(min(sims), 4),
            "max_sim"             : round(max(sims), 4),
            "most_different_pair" : (pairs[0][0],  pairs[0][1]),
            "most_similar_pair"   : (pairs[-1][0], pairs[-1][1]),
            "total_updates"       : self.total_updates,
        }

    def snapshot(self) -> dict:
        return {
            "total_updates" : self.total_updates,
            "total_routes"  : self.total_routes,
            "attractors"    : list(self._attractors.keys()),
            "cluster"       : self.cluster_summary(),
        }


if __name__ == "__main__":
    print("=" * 56)
    print("PANTHEON Week 3 — SpeciationEngine smoke test")
    print("=" * 56)

    TASK_TYPES = ["code", "research", "visual"]
    N_AGENTS   = 10
    N_TASKS    = 600

    engine = SpeciationEngine(n_agents=N_AGENTS, alpha_scale=0.05,
                              top_n=3, seed=42)

    print("\nSeed separations:")
    names = list(engine._type_seeds.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            sim = torch.dot(engine._type_seeds[names[i]],
                            engine._type_seeds[names[j]]).item()
            print(f"  {names[i]} vs {names[j]}: {sim:.3f}")

    bias = {}
    for i in range(N_AGENTS):
        if i < 3:
            bias[i] = {"code": 0.85, "research": 0.15, "visual": 0.15}
        elif i < 6:
            bias[i] = {"code": 0.15, "research": 0.85, "visual": 0.15}
        else:
            bias[i] = {"code": 0.15, "research": 0.15, "visual": 0.85}

    task_descs = {
        "code"    : "implement a sorting algorithm in python",
        "research": "write a hypothesis about emergent agent behavior",
        "visual"  : "generate a t-SNE plot of agent fingerprints",
    }

    print(f"\nSimulating {N_TASKS} tasks...")
    for tick in range(N_TASKS):
        tt      = random.choice(TASK_TYPES)
        aid     = random.randint(0, N_AGENTS - 1)
        success = random.random() < bias[aid][tt]
        reward  = random.randint(5, 18) if success else 0
        engine.update_fingerprint(aid, tt, success, reward, tick)

        if (tick + 1) % 100 == 0:
            s = engine.cluster_summary()
            print(f"  tick {tick+1:4d}  "
                  f"mean_sim={s['mean_pairwise_sim']:.4f}  "
                  f"min_sim={s['min_sim']:.4f}")

    final = engine.cluster_summary()
    print(f"\nFinal:")
    for k, v in final.items():
        print(f"  {k}: {v}")

    diverged  = final["mean_pairwise_sim"] < 0.85
    separated = final["min_sim"] < 0.3

    print(f"\n{'[PASS]' if diverged  else '[FAIL]'} diverged   (mean={final['mean_pairwise_sim']})")
    print(f"{'[PASS]' if separated else '[FAIL]'} separated  (min={final['min_sim']})")

    print("\nRouting test:")
    living = list(range(N_AGENTS))
    for tt in TASK_TYPES:
        routed = engine.route_task(task_descs[tt], tt, living)
        print(f"  {tt:8s} -> agent {routed}  bias={bias[routed][tt]:.2f}")

    print("\n" + "=" * 56)
    print("Week 3 Piece 1 — DONE")
    print("=" * 56)