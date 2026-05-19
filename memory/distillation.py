from __future__ import annotations
 
import threading
import logging
import numpy as np
import networkx as nx
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
 
try:
    from sklearn.cluster import KMeans
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
 
from memory.episodic import EpisodicMemory, EpisodicRecord, DIM
 
logger = logging.getLogger(__name__)
 
N_CLUSTERS = 20          # matches plan
EDGE_THRESHOLD = 0.7     # cosine similarity floor for graph edges
EDGE_DECAY = 0.001       # per tick
EDGE_BOOST = 0.1         # on retrieval hit
MIN_RECORDS = 100        # don't distill below this
 
 
@dataclass
class SemanticNode:
    node_id: int
    embedding: np.ndarray   # cluster centroid, 128-dim
    label: str              # top tokens from member records
    quality: float          # mean quality of cluster members
    frequency: int          # number of source records
    member_task_ids: list[str] = field(default_factory=list)
 
 
class KnowledgeDistiller:
    """
    Compresses episodic memory into a semantic knowledge graph.
 
    Graph contract:
        nodes: SemanticNode (embedding, label, quality, frequency)
        edges: {weight: float}  — cosine similarity, Hebbian-modulated
 
    RAG contract:
        .retrieve(task_emb, k) → list[SemanticNode]
        Called by agents before tackling any task (via SharedMemory bridge).
 
    Distillation contract:
        .distill() is called by LibrarianAgent every 100 ticks on background thread.
        Never blocks the main agent loop.
    """
 
    def __init__(
        self,
        episodic: EpisodicMemory,
        db_path: str = "artifacts/artifacts.db",
        n_clusters: int = N_CLUSTERS,
    ):
        self.episodic = episodic
        self.n_clusters = n_clusters
        self._graph = nx.Graph()
        self._nodes: dict[int, SemanticNode] = {}
        self._lock = threading.Lock()
        self._db = self._init_db(db_path)
 
    # ------------------------------------------------------------------ #
    # Core distillation — runs on background thread                       #
    # ------------------------------------------------------------------ #
 
    def distill(self) -> int:
        """
        Run one distillation pass. Returns number of nodes created/updated.
        Safe to call concurrently — holds internal lock only during graph write.
        """
        records = self.episodic.all_records()
        if len(records) < MIN_RECORDS:
            logger.debug(f"Distillation skipped: only {len(records)} records (need {MIN_RECORDS})")
            return 0
 
        vecs = np.stack([r.solution_emb for r in records]).astype(np.float32)
        labels = self._cluster(vecs)
 
        new_nodes: dict[int, SemanticNode] = {}
        for cluster_id in range(self.n_clusters):
            members = [r for r, l in zip(records, labels) if l == cluster_id]
            if not members:
                continue
 
            centroid = vecs[labels == cluster_id].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            label = self._top_tokens(members)
            quality = float(np.mean([m.quality for m in members]))
            task_ids = [m.task_id for m in members]
 
            new_nodes[cluster_id] = SemanticNode(
                node_id=cluster_id,
                embedding=centroid,
                label=label,
                quality=quality,
                frequency=len(members),
                member_task_ids=task_ids,
            )
 
        with self._lock:
            self._nodes = new_nodes
            self._rebuild_graph()
            self._persist_nodes()
 
        logger.info(f"Distillation complete: {len(new_nodes)} semantic nodes, "
                    f"{self._graph.number_of_edges()} edges")
        return len(new_nodes)
 
    # ------------------------------------------------------------------ #
    # Retrieval (RAG path)                                                 #
    # ------------------------------------------------------------------ #
 
    def retrieve(self, task_emb: np.ndarray, k: int = 3) -> list[SemanticNode]:
        """
        Return k nearest semantic nodes by cosine similarity to task_emb.
        Boosts edge weights for retrieved nodes (Hebbian reinforcement).
        """
        task_emb = task_emb.astype(np.float32)
        task_emb /= np.linalg.norm(task_emb) + 1e-8
 
        with self._lock:
            if not self._nodes:
                return []
 
            centroids = np.stack([n.embedding for n in self._nodes.values()])
            node_ids = list(self._nodes.keys())
            sims = centroids @ task_emb
            top_k_idx = np.argsort(sims)[::-1][:k]
 
            results = []
            for idx in top_k_idx:
                nid = node_ids[idx]
                node = self._nodes[nid]
                results.append(node)
                # Hebbian boost on connected edges
                for neighbor in self._graph.neighbors(nid):
                    if self._graph.has_edge(nid, neighbor):
                        self._graph[nid][neighbor]['weight'] = min(
                            1.0,
                            self._graph[nid][neighbor]['weight'] + EDGE_BOOST
                        )
 
        return results
 
    # ------------------------------------------------------------------ #
    # Hebbian tick — called from main loop                                #
    # ------------------------------------------------------------------ #
 
    def tick_decay(self) -> None:
        """Decay all edge weights by EDGE_DECAY. Dead edges (≤0) are removed."""
        with self._lock:
            dead = [
                (u, v) for u, v, d in self._graph.edges(data=True)
                if d.get('weight', 0) - EDGE_DECAY <= 0
            ]
            for u, v in dead:
                self._graph.remove_edge(u, v)
            for u, v in self._graph.edges():
                self._graph[u][v]['weight'] = max(
                    0.0, self._graph[u][v]['weight'] - EDGE_DECAY
                )
 
    # ------------------------------------------------------------------ #
    # Graph accessors                                                      #
    # ------------------------------------------------------------------ #
 
    @property
    def graph(self) -> nx.Graph:
        return self._graph
 
    @property
    def node_count(self) -> int:
        return len(self._nodes)
 
    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
 
    def get_node(self, node_id: int) -> Optional[SemanticNode]:
        return self._nodes.get(node_id)
 
    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #
 
    def _cluster(self, vecs: np.ndarray) -> np.ndarray:
        k = min(self.n_clusters, len(vecs))
        if _SKLEARN:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            return km.fit_predict(vecs)
        else:
            # numpy fallback: random init k-means, 10 iterations
            centres = vecs[np.random.choice(len(vecs), k, replace=False)]
            labels = np.zeros(len(vecs), dtype=int)
            for _ in range(10):
                dists = np.linalg.norm(vecs[:, None] - centres[None], axis=2)
                labels = dists.argmin(axis=1)
                for i in range(k):
                    mask = labels == i
                    if mask.any():
                        centres[i] = vecs[mask].mean(axis=0)
            return labels
 
    def _top_tokens(self, members: list[EpisodicRecord], n: int = 5) -> str:
        """
        Proxy for 'top tokens': use task_id prefixes as a label.
        In production, replace with actual token frequency from agent vocab.
        """
        task_types = [m.task_id.split(":")[0] for m in members]
        counter = Counter(task_types)
        return ", ".join(f"{t}({c})" for t, c in counter.most_common(n))
 
    def _rebuild_graph(self) -> None:
        """Rebuild edges from scratch using cosine similarity threshold."""
        self._graph.clear()
        node_ids = list(self._nodes.keys())
        for nid, node in self._nodes.items():
            self._graph.add_node(
                nid,
                label=node.label,
                quality=node.quality,
                frequency=node.frequency,
            )
        # Add edges where cosine similarity > threshold
        if len(node_ids) > 1:
            centroids = np.stack([self._nodes[n].embedding for n in node_ids])
            sim_matrix = centroids @ centroids.T  # cosine (already normed)
            for i, u in enumerate(node_ids):
                for j, v in enumerate(node_ids):
                    if i >= j:
                        continue
                    sim = float(sim_matrix[i, j])
                    if sim > EDGE_THRESHOLD:
                        self._graph.add_edge(u, v, weight=sim)
 
    def _init_db(self, db_path: str) -> sqlite3.Connection:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_nodes (
                node_id   INTEGER PRIMARY KEY,
                label     TEXT,
                quality   REAL,
                frequency INTEGER,
                timestamp REAL
            )
        """)
        conn.commit()
        return conn
 
    def _persist_nodes(self) -> None:
        import time
        rows = [
            (n.node_id, n.label, n.quality, n.frequency, time.time())
            for n in self._nodes.values()
        ]
        self._db.executemany(
            "INSERT OR REPLACE INTO semantic_nodes VALUES (?,?,?,?,?)", rows
        )
        self._db.commit()
 