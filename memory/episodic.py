from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import sqlite3
import threading
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

DIM = 128


@dataclass
class EpisodicRecord:
    record_id    : int
    task_id      : str
    task_emb     : np.ndarray
    solution_emb : np.ndarray
    quality      : float
    agent_id     : int
    coalition_id : Optional[str]
    timestamp    : float = field(default_factory=time.time)
    decay_weight : float = 1.0


class EpisodicMemory:
    DECAY_RATE      = 0.001
    RETRIEVAL_BOOST = 0.1
    QUALITY_FLOOR   = 0.3

    def __init__(self, db_path: str = "artifacts/episodic.db", dim: int = DIM):
        self.dim      = dim
        self.db_path  = db_path
        self._lock    = threading.Lock()
        self._records : list[EpisodicRecord] = []
        self._next_id = 0

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatL2(dim)
        else:
            self._index = None

        self._db = self._init_db()

    def record(self,
               task_id      : str,
               task_emb     : np.ndarray,
               solution_emb : np.ndarray,
               quality      : float,
               agent_id     : int,
               coalition_id : Optional[str] = None) -> EpisodicRecord:
        task_emb     = _norm(task_emb)
        solution_emb = _norm(solution_emb)

        with self._lock:
            rec = EpisodicRecord(
                record_id    = self._next_id,
                task_id      = task_id,
                task_emb     = task_emb,
                solution_emb = solution_emb,
                quality      = quality,
                agent_id     = agent_id,
                coalition_id = coalition_id,
            )
            self._next_id += 1
            self._records.append(rec)
            self._faiss_add(task_emb)
            self._db_insert(rec)

        return rec

    def retrieve(self, task_emb: np.ndarray, k: int = 3) -> list[EpisodicRecord]:
        task_emb = _norm(task_emb)

        with self._lock:
            if not self._records:
                return []

            indices = self._faiss_search(task_emb, k)
            results = []
            for i in indices:
                if 0 <= i < len(self._records):
                    rec = self._records[i]
                    rec.decay_weight = min(1.0, rec.decay_weight + self.RETRIEVAL_BOOST)
                    results.append(rec)

        return results

    def tick_decay(self) -> None:
        with self._lock:
            for rec in self._records:
                shield          = rec.quality
                effective_decay = self.DECAY_RATE * (2.0 - shield)
                rec.decay_weight = max(0.0, rec.decay_weight - effective_decay)

    def compress(self) -> int:
        with self._lock:
            before = len(self._records)
            self._records = [
                r for r in self._records
                if not (r.quality < self.QUALITY_FLOOR and r.decay_weight < 0.1)
            ]
            pruned = before - len(self._records)
            if pruned > 0:
                self._rebuild_faiss()
            return pruned

    def all_records(self) -> list[EpisodicRecord]:
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def _faiss_add(self, vec: np.ndarray) -> None:
        if self._index is not None:
            self._index.add(vec.reshape(1, -1).astype(np.float32))

    def _faiss_search(self, vec: np.ndarray, k: int) -> list[int]:
        k = min(k, len(self._records))
        if self._index is not None and self._index.ntotal > 0:
            _, indices = self._index.search(
                vec.reshape(1, -1).astype(np.float32), k
            )
            return indices[0].tolist()
        mat   = np.stack([r.task_emb for r in self._records])
        dists = np.linalg.norm(mat - vec, axis=1)
        return np.argsort(dists)[:k].tolist()

    def _rebuild_faiss(self) -> None:
        if self._index is not None:
            self._index.reset()
            if self._records:
                mat = np.stack([r.task_emb for r in self._records]).astype(np.float32)
                self._index.add(mat)

    def _init_db(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                record_id    INTEGER PRIMARY KEY,
                task_id      TEXT,
                quality      REAL,
                agent_id     INTEGER,
                coalition_id TEXT,
                timestamp    REAL,
                decay_weight REAL
            )
        """)
        conn.commit()
        return conn

    def _db_insert(self, rec: EpisodicRecord) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO episodic_memory VALUES (?,?,?,?,?,?,?)",
            (rec.record_id, rec.task_id, rec.quality,
             rec.agent_id, rec.coalition_id, rec.timestamp, rec.decay_weight),
        )
        self._db.commit()


def _norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v
