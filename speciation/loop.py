import random
import uuid
import torch
import torch.nn.functional as F

from core.communication_bus   import CommunicationBus, Signal, Tag
from substrate.registry        import AgentRegistry
from substrate.economy         import ComputeEconomy
from core.taskqueue            import TaskQueue, TaskType, TaskStatus, Task
from substrate.artifact_store  import ArtifactStore
from speciation.engine         import SpeciationEngine

try:
    from speciation.train_encoder import train_encoder
except ModuleNotFoundError:
    from train_encoder import train_encoder


class SyntheticAgent:
    def __init__(self, agent_id, task_type_bias=None):
        self.agent_id = agent_id
        self.bias = task_type_bias or {
            "code"    : random.uniform(0.3, 0.7),
            "research": random.uniform(0.3, 0.7),
            "visual"  : random.uniform(0.3, 0.7),
        }

    def attempt(self, task):
        p       = self.bias[task.task_type.value]
        success = random.random() < p
        quality = random.uniform(0.5, 1.0) * p if success else 0.0
        content = (f"# agent={self.agent_id} type={task.task_type.value} "
                   f"quality={quality:.2f}\nresult = '{uuid.uuid4().hex[:8]}'")
        return success, quality, content


TASK_BANK = {
    "code": [
        "implement binary search in python",
        "write a quicksort algorithm",
        "build a graph traversal function",
        "code a hash map implementation",
        "implement a recursive fibonacci",
        "write a linked list from scratch",
        "build a stack and queue data structure",
        "implement an LRU cache",
    ],
    "research": [
        "write a hypothesis about emergent agent behavior",
        "propose a falsifiable claim about token economies",
        "formulate a research question on collective intelligence",
        "design an experiment to test agent specialization",
        "write a hypothesis about coalition formation dynamics",
        "propose a study on reputation systems in agents",
        "formulate a claim about evolutionary pressure in AI",
        "design a benchmark for collective problem solving",
    ],
    "visual": [
        "generate a t-SNE plot of agent fingerprints",
        "create a force-directed graph of active coalitions",
        "plot token balance distribution over time",
        "visualize the knowledge graph as a network diagram",
        "generate a heatmap of agent task success rates",
        "create a timeline of agent births and deaths",
        "plot the loss curve of the society transformer",
        "visualize fingerprint cluster evolution over ticks",
    ],
}


def sample_task(task_type_str: str, tick: int, queue: TaskQueue) -> Task:
    """Post one typed task from the task bank."""
    desc = random.choice(TASK_BANK[task_type_str])
    diff = random.uniform(0.3, 1.0)
    tt   = TaskType(task_type_str)
    return queue.post(tt, diff, desc, tick)


def count_clusters(engine: SpeciationEngine, threshold: float = 0.5) -> int:
    """
    Greedy clustering: assign each agent to the first existing cluster
    whose centroid has cosine sim > threshold with the agent's fp.
    Returns number of clusters found.
    """
    fps      = engine.all_fingerprints()
    clusters = []

    for i in range(fps.shape[0]):
        fp       = fps[i]
        assigned = False
        for j, centroid in enumerate(clusters):
            if torch.dot(fp, centroid).item() > threshold:
                clusters[j] = F.normalize(centroid + fp, dim=0)
                assigned = True
                break
        if not assigned:
            clusters.append(F.normalize(fp.clone(), dim=0))

    return len(clusters)


def print_distance_matrix(engine: SpeciationEngine):
    fps = engine.all_fingerprints()
    n   = fps.shape[0]
    print("\n  Pairwise cosine similarity matrix:")
    print("       " + "  ".join(f"A{i}" for i in range(n)))
    for i in range(n):
        row = [f"{torch.dot(fps[i], fps[j]).item():+.2f}" for j in range(n)]
        print(f"  A{i}  " + "  ".join(row))



class Week3Society:
    TASK_TYPES = ["code", "research", "visual"]

    def __init__(self, n_agents: int = 10, seed: int = 42):
        random.seed(seed)
        torch.manual_seed(seed)

        self.n_agents = n_agents

        self.registry = AgentRegistry(n_agents)
        self.economy  = ComputeEconomy(n_agents)
        self.queue    = TaskQueue(expiry_ticks=50)

        self.engine = SpeciationEngine(
            n_agents    = n_agents,
            alpha_scale = 0.05,
            top_n       = 3,
            seed        = seed,
        )

        self.agents = {}
        for i in range(n_agents):
            if i < 3:
                bias = {"code": 0.82, "research": 0.18, "visual": 0.18}
            elif i < 6:
                bias = {"code": 0.18, "research": 0.82, "visual": 0.18}
            else:
                bias = {"code": 0.18, "research": 0.18, "visual": 0.82}
            self.agents[i] = SyntheticAgent(i, bias)

        # Fixed seed vectors for encoder training (must be outside the loop)
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.type_seeds = {}
        for name in self.TASK_TYPES:
            raw = torch.randn(128, generator=gen)
            self.type_seeds[name] = F.normalize(raw, dim=0)

        self.tick = 0
        self._routing_correct = []

    def setup(self):
        print("Training TaskEncoder...")
        trained_encoder = train_encoder(
            self.type_seeds, steps=300, lr=3e-3,
            batch_size=8, verbose=False
        )
        self.engine.encoder = trained_encoder
        print("TaskEncoder trained\n")

        self.economy.death_enabled = True
        print("Death enabled\n")

    def run(self, ticks: int = 500):
        print(f"Running {ticks} ticks, {self.n_agents} agents...\n")

        for t in range(ticks):
            self.tick = t
            self._tick(t)

            if (t + 1) % 100 == 0:
                self._report(t + 1)
                self.engine.snapshot_fingerprints(t + 1)

        self._final_report(ticks)

    def _tick(self, t: int):
        posted_tasks = []
        for tt in self.TASK_TYPES:
            task = sample_task(tt, t, self.queue)
            posted_tasks.append(task)

        living = list(self.registry.records.keys())

        for _ in posted_tasks:
            popped = self.queue.dispatch(0, t)   # placeholder agent_id
            if popped is None:
                continue

            best_agent = self.engine.route_task(
                popped.description,
                popped.task_type.value,
                living,
            )
            popped.assigned_agent = best_agent

            tt_val  = popped.task_type.value
            agent   = self.agents[best_agent]
            correct = agent.bias[tt_val] > 0.5
            self._routing_correct.append(correct)

            success, quality, content = agent.attempt(popped)

            if success:
                done = self.queue.complete(popped.task_id, t)
                if done is None:
                    continue
                self.economy.earn(best_agent, done.reward_tokens)
                self.registry.update_reputation(best_agent, quality)
                self.engine.update_fingerprint(
                    agent_id  = best_agent,
                    task_type = tt_val,
                    success   = True,
                    reward    = done.reward_tokens,
                    tick      = t,
                )
            else:
                failed = self.queue.fail(popped.task_id, t)
                if failed is None:
                    continue
                self.economy.spend(best_agent, failed.penalty_tokens)
                self.engine.update_fingerprint(
                    agent_id  = best_agent,
                    task_type = tt_val,
                    success   = False,
                    reward    = 0,
                    tick      = t,
                )

        # Check for deaths (one tick() call per agent)
        for aid in list(self.registry.records.keys()):
            event = self.economy.tick(aid)
            if event:
                dead, parent = event["killed"], event["parent"]
                self.engine.replace_agent(dead, parent, current_tick=t)
                self.registry.replace_agent(dead, parent, current_tick=t)
                print(f"  [DEATH] tick={t} agent={dead} → born from agent={parent}")

        self.engine.sync_to_registry(self.registry)

    def _report(self, t: int):
        cluster    = self.engine.cluster_summary()
        n_clust    = count_clusters(self.engine, threshold=0.5)
        q_snap     = self.queue.snapshot()
        bal        = self.economy.snapshot()
        recent_acc = (sum(self._routing_correct[-300:]) /
                      max(1, len(self._routing_correct[-300:])))

        print(f"── tick {t:4d} " + "─" * 36)
        print(f"  clusters     : {n_clust}  (need ≥3 for success)")
        print(f"  mean_sim     : {cluster.get('mean_pairwise_sim', '?')}")
        print(f"  min_sim      : {cluster.get('min_sim', '?')}")
        print(f"  routing acc  : {recent_acc:.1%}  "
              f"(specialists getting specialist tasks)")
        print(f"  tasks        : {q_snap}")
        print(f"  token range  : min={min(bal.values())}  max={max(bal.values())}")
        print(f"  attractors   : {list(self.engine._attractors.keys())}")
        print()

    def _final_report(self, ticks: int):
        print("═" * 52)
        print("WEEK 3 FINAL REPORT")
        print("═" * 52)

        cluster  = self.engine.cluster_summary()
        n_clust  = count_clusters(self.engine, threshold=0.5)
        living   = self.registry.all_living()
        q_snap   = self.queue.snapshot()
        acc      = (sum(self._routing_correct) /
                    max(1, len(self._routing_correct)))

        mean_sim = cluster.get("mean_pairwise_sim", 1.0)
        min_sim  = cluster.get("min_sim", 1.0)
        c1 = len(living) >= 8
        c2 = mean_sim < 0.85
        c3 = n_clust >= 3
        c4 = acc > 0.5

        print(f"\n{'✓' if c1 else '✗'} agents alive          : {len(living)}/10")
        print(f"{'✓' if c2 else '✗'} fingerprints diverged : mean_sim={mean_sim}")
        print(f"{'✓' if c3 else '✗'} clusters ≥ 3          : {n_clust} found")
        print(f"{'✓' if c4 else '✗'} routing accuracy      : {acc:.1%}")

        print_distance_matrix(self.engine)

        print(f"\nAttractors formed: {list(self.engine._attractors.keys())}")
        print(f"Total fingerprint updates: {self.engine.total_updates}")

        print(f"\nToken balances:")
        for aid, bal in sorted(self.economy.snapshot().items()):
            bar    = "█" * max(0, bal // 15)
            marker = ("CODE" if aid < 3 else
                      "RESEARCH" if aid < 6 else "VISUAL")
            print(f"  A{aid} [{marker:8s}]: {bal:5d}  {bar}")

        all_pass = c1 and c2 and c3 and c4
        print(f"\n{'Done' if all_pass else '⚠ Some criteria not met'}")
        print("═" * 52)




if __name__ == "__main__":
    print("=" * 52)
    print("PANTHEON Week 3 — Speciation Engine Loop")
    print("=" * 52 + "\n")

    society = Week3Society(n_agents=10, seed=42)
    society.setup()
    society.run(ticks=500)
