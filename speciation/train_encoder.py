import random
import torch
import torch.nn.functional as F
try:
    from speciation.engine import TaskEncoder
except ModuleNotFoundError:
    from engine import TaskEncoder

TASK_POOL = {
    "code": [
        "implement binary search in python",
        "write a bubble sort algorithm",
        "build a linked list from scratch",
        "code a depth-first graph traversal",
        "implement a hash map with collision handling",
        "write a recursive fibonacci function",
        "build a stack using arrays",
        "implement quicksort in python",
        "write a function to detect palindromes",
        "code a breadth-first search algorithm",
        "implement a min heap data structure",
        "write a tokenizer for simple expressions",
        "build a basic calculator with parsing",
        "implement LRU cache in python",
        "write a function to flatten nested lists",
    ],
    "research": [
        "write a hypothesis about emergent agent behavior",
        "propose a falsifiable claim about token economies",
        "formulate a research question on collective intelligence",
        "design an experiment to test agent specialization",
        "write a literature review on multi-agent systems",
        "propose a hypothesis about coalition formation",
        "formulate a claim about evolutionary pressure in AI",
        "design a study on reputation systems in agents",
        "write a hypothesis about knowledge distillation",
        "propose an experiment on agent memory retrieval",
        "formulate a claim about self-modeling in societies",
        "design a benchmark for collective problem solving",
        "write a hypothesis about role emergence",
        "propose a study on agent communication protocols",
        "formulate a research question on agent diversity",
    ],
    "visual": [
        "generate a t-SNE plot of agent fingerprints",
        "create a force-directed graph of coalitions",
        "plot token balance distribution over time",
        "visualize the knowledge graph as a network",
        "generate a heatmap of agent task success rates",
        "create a timeline of agent births and deaths",
        "plot the loss curve of the society model",
        "visualize agent reputation scores as a bar chart",
        "generate a scatter plot of fingerprint clusters",
        "create a sankey diagram of task routing",
        "plot the evolution of skill attractors over ticks",
        "visualize artifact quality scores by agent",
        "generate a histogram of coalition sizes",
        "create a flow diagram of the agent lifecycle",
        "plot pairwise fingerprint similarities as a matrix",
    ],
}

TASK_TYPES = ["code", "research", "visual"]


def contrastive_loss(encoder     : TaskEncoder,
                     type_seeds  : dict,
                     batch_size  : int   = 8,
                     margin      : float = 0.5,
                     seed_weight : float = 0.6):
    seed_losses = []
    same_losses = []
    diff_losses = []

    all_type_embs = {}
    for task_type in TASK_TYPES:
        tasks = random.sample(TASK_POOL[task_type], min(batch_size, len(TASK_POOL[task_type])))
        embs  = torch.stack([encoder(t) for t in tasks])
        all_type_embs[task_type] = F.normalize(embs, dim=1)

        seed     = type_seeds[task_type]
        seed_sim = (all_type_embs[task_type] * seed).sum(dim=1)
        seed_losses.append((1.0 - seed_sim).mean())

        sim  = all_type_embs[task_type] @ all_type_embs[task_type].T
        mask = torch.ones_like(sim).triu(diagonal=1).bool()
        if mask.any():
            same_losses.append((1.0 - sim[mask]).mean())

    for i, ti in enumerate(TASK_TYPES):
        for j, tj in enumerate(TASK_TYPES):
            if i >= j:
                continue
            cross = all_type_embs[ti] @ all_type_embs[tj].T
            diff_losses.append(F.relu(cross - margin).mean())

    L_seed = torch.stack(seed_losses).mean()
    L_same = torch.stack(same_losses).mean() if same_losses else torch.tensor(0.0)
    L_diff = torch.stack(diff_losses).mean() if diff_losses else torch.tensor(0.0)

    total = seed_weight * L_seed + 0.3 * L_same + 0.1 * L_diff
    return total, L_seed.item(), L_same.item(), L_diff.item()


def train_encoder(type_seeds : dict,
                  steps      : int   = 300,
                  lr         : float = 3e-3,
                  batch_size : int   = 8,
                  verbose    : bool  = True) -> TaskEncoder:
    encoder   = TaskEncoder()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05
    )
    encoder.train()

    for step in range(steps):
        loss, l_seed, l_same, l_diff = contrastive_loss(
            encoder, type_seeds, batch_size=batch_size
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if verbose and step % 50 == 0:
            print(f"  step {step:4d}  "
                  f"loss={loss.item():.4f}  "
                  f"seed={l_seed:.4f}  "
                  f"same={l_same:.4f}  "
                  f"diff={l_diff:.4f}")

    encoder.eval()
    return encoder


def evaluate_encoder(encoder : TaskEncoder, type_seeds : dict) -> dict:
    encoder.eval()
    results  = {}
    all_embs = {}

    with torch.no_grad():
        for task_type, tasks in TASK_POOL.items():
            all_embs[task_type] = F.normalize(
                torch.stack([encoder(t) for t in tasks]), dim=1
            )

        same_sims = []
        for task_type in TASK_TYPES:
            embs = all_embs[task_type]
            sim  = embs @ embs.T
            mask = torch.ones_like(sim).triu(diagonal=1).bool()
            same_sims.extend(sim[mask].tolist())
        results["mean_same_type_sim"] = round(sum(same_sims) / len(same_sims), 4)

        diff_sims = []
        for i, ti in enumerate(TASK_TYPES):
            for j, tj in enumerate(TASK_TYPES):
                if i >= j:
                    continue
                diff_sims.extend(
                    (all_embs[ti] @ all_embs[tj].T).flatten().tolist()
                )
        results["mean_diff_type_sim"] = round(sum(diff_sims) / len(diff_sims), 4)

        seed_sims = {}
        for task_type in TASK_TYPES:
            embs = all_embs[task_type]
            seed = type_seeds[task_type]
            sim  = (embs * seed).sum(dim=1).mean().item()
            seed_sims[task_type] = round(sim, 4)
        results["seed_alignment"] = seed_sims

        if results["mean_diff_type_sim"] > 0:
            results["separation_ratio"] = round(
                results["mean_same_type_sim"] / results["mean_diff_type_sim"], 3
            )
        else:
            results["separation_ratio"] = 999.0

    return results


if __name__ == "__main__":
    print("=" * 56)
    print("PANTHEON Week 3 — TaskEncoder training")
    print("=" * 56)

    gen = torch.Generator()
    gen.manual_seed(42)
    type_seeds = {}
    for name in TASK_TYPES:
        raw = torch.randn(128, generator=gen)
        type_seeds[name] = F.normalize(raw, dim=0)

    print("\nBefore training:")
    untrained = TaskEncoder()
    before = evaluate_encoder(untrained, type_seeds)
    for k, v in before.items():
        print(f"  {k}: {v}")

    print(f"\nTraining 300 steps...")
    trained = train_encoder(type_seeds, steps=300, lr=3e-3, batch_size=8)

    print("\nAfter training:")
    after = evaluate_encoder(trained, type_seeds)
    for k, v in after.items():
        print(f"  {k}: {v}")

    print()
    same_improved = after["mean_same_type_sim"] > before["mean_same_type_sim"]
    diff_dropped  = after["mean_diff_type_sim"]  < before["mean_diff_type_sim"]
    separated     = after["separation_ratio"]     > 1.5

    print(f"{'[PASS]' if same_improved else '[FAIL]'} "
          f"same-type sim improved  "
          f"({before['mean_same_type_sim']} -> {after['mean_same_type_sim']})")
    print(f"{'[PASS]' if diff_dropped  else '[FAIL]'} "
          f"diff-type sim dropped   "
          f"({before['mean_diff_type_sim']} -> {after['mean_diff_type_sim']})")
    print(f"{'[PASS]' if separated     else '[FAIL]'} "
          f"separation ratio > 1.5  ({after['separation_ratio']})")

    torch.save(trained.state_dict(), "speciation/encoder_weights.pt")
    print(f"\nSaved to speciation/encoder_weights.pt")

    print("\n" + "=" * 56)
    print("TaskEncoder training — DONE")
    print("=" * 56)
