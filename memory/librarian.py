from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from memory.distillation import KnowledgeDistiller
from memory.episodic     import EpisodicMemory

logger = logging.getLogger(__name__)

DISTILL_INTERVAL      = 100
PRESTIGE_BONUS        = 20
LIBRARIAN_SEEDED      = True
_LIBRARIAN_SINGLETON  : Optional["LibrarianAgent"] = None


class BaseCoherenceScorer(ABC):
    @abstractmethod
    def score(self, hypothesis: dict) -> float: ...


class MockCoherenceScorer(BaseCoherenceScorer):
    REQUIRED_KEYS    = {"claim", "evidence_needed", "experiment", "falsifiable"}
    QUALITY_KEYWORDS = {
        "because", "therefore", "hypothesis", "measure", "predict",
        "data", "result", "correlation", "test", "significant",
    }

    def score(self, hypothesis: dict) -> float:
        if not isinstance(hypothesis, dict):
            return 0.0
        key_score = len(self.REQUIRED_KEYS & hypothesis.keys()) / len(self.REQUIRED_KEYS)
        claim     = str(hypothesis.get("claim", "")).lower().split()
        if not claim:
            return key_score * 0.5
        kw_hits       = sum(1 for w in claim if w in self.QUALITY_KEYWORDS)
        content_score = min(1.0, kw_hits / max(1, len(claim)) * 10)
        falsifiable_bonus = 0.1 if hypothesis.get("falsifiable") is True else 0.0
        return min(1.0, 0.5 * key_score + 0.4 * content_score + falsifiable_bonus)


class MlxCoherenceScorer(BaseCoherenceScorer):
    MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

    def __init__(self):
        from mlx_lm import load, generate  # type: ignore
        self._model, self._tokenizer = load(self.MODEL)
        self._generate = generate
        logger.info("MlxCoherenceScorer loaded: %s", self.MODEL)

    def score(self, hypothesis: dict) -> float:
        prompt = (
            f"Rate this research hypothesis for scientific coherence from 0.0 to 1.0. "
            f"Reply with only a float number.\n\nHypothesis: {hypothesis}\n\nScore:"
        )
        response = self._generate(self._model, self._tokenizer,
                                  prompt=prompt, max_tokens=5)
        try:
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            return 0.5


class LlamaCppCoherenceScorer(BaseCoherenceScorer):
    def __init__(self, model_path: str):
        from llama_cpp import Llama  # type: ignore
        self._llm = Llama(model_path=model_path, n_ctx=256, verbose=False)
        logger.info("LlamaCppCoherenceScorer loaded: %s", model_path)

    def score(self, hypothesis: dict) -> float:
        prompt = (
            f"Rate coherence 0.0-1.0, reply float only.\n"
            f"Hypothesis: {hypothesis}\nScore:"
        )
        out = self._llm(prompt, max_tokens=5)
        try:
            return max(0.0, min(1.0, float(out["choices"][0]["text"].strip())))
        except (ValueError, KeyError):
            return 0.5


def build_scorer(scorer_type: Optional[str] = None) -> BaseCoherenceScorer:
    kind = scorer_type or os.environ.get("SCORER", "mock")
    if kind == "mlx":
        return MlxCoherenceScorer()
    elif kind == "llama":
        model_path = os.environ.get("MODEL_PATH", "models/llama-3.2-1b.gguf")
        return LlamaCppCoherenceScorer(model_path)
    logger.info("Using MockCoherenceScorer (set SCORER=mlx or SCORER=llama to upgrade)")
    return MockCoherenceScorer()


@dataclass
class DistillationReport:
    tick           : int
    nodes_created  : int
    edge_count     : int
    records_pruned : int
    duration_s     : float


class LibrarianAgent:
    AGENT_ID = -1

    def __init__(self,
                 episodic         : EpisodicMemory,
                 distiller        : KnowledgeDistiller,
                 economy          = None,
                 bus              = None,
                 scorer           : Optional[BaseCoherenceScorer] = None,
                 distill_interval : int  = DISTILL_INTERVAL,
                 seeded           : bool = LIBRARIAN_SEEDED):
        self.episodic         = episodic
        self.distiller        = distiller
        self.economy          = economy
        self.bus              = bus
        self.scorer           = scorer or build_scorer()
        self.distill_interval = distill_interval
        self.seeded           = seeded

        self._last_distill_tick = -distill_interval
        self._bg_thread         : Optional[threading.Thread] = None
        self._running           = False
        self.reports            : list[DistillationReport] = []

        logger.info("LibrarianAgent initialised (agent_id=%d, seeded=%s, scorer=%s)",
                    self.AGENT_ID, seeded, type(self.scorer).__name__)

    def on_tick(self, tick: int) -> None:
        self.episodic.tick_decay()
        self.distiller.tick_decay()

        if tick - self._last_distill_tick >= self.distill_interval:
            if self._bg_thread is None or not self._bg_thread.is_alive():
                self._last_distill_tick = tick
                self._bg_thread = threading.Thread(
                    target=self._distill_job, args=(tick,), daemon=True
                )
                self._bg_thread.start()

    def _distill_job(self, tick: int) -> None:
        t0 = time.perf_counter()
        try:
            nodes_created = self.distiller.distill()
            pruned        = self.episodic.compress()
            elapsed       = time.perf_counter() - t0

            report = DistillationReport(
                tick           = tick,
                nodes_created  = nodes_created,
                edge_count     = self.distiller.edge_count,
                records_pruned = pruned,
                duration_s     = elapsed,
            )
            self.reports.append(report)

            if self.economy is not None:
                self.economy.earn(self.AGENT_ID, PRESTIGE_BONUS)

            if self.bus is not None:
                self.bus.publish(
                    tag       = "distillation_complete",
                    payload   = report,
                    sender_id = self.AGENT_ID,
                )

            logger.info("Distillation @tick=%d: %d nodes, %d edges, %d pruned, %.2fs",
                        tick, nodes_created, report.edge_count, pruned, elapsed)
        except Exception as exc:
            logger.error("Distillation job failed: %s", exc, exc_info=True)

    def score_hypothesis(self, hypothesis: dict) -> float:
        return self.scorer.score(hypothesis)

    def retrieve_context(self, task_emb, k: int = 3):
        episodic = self.episodic.retrieve(task_emb, k=k)
        semantic = self.distiller.retrieve(task_emb, k=k)
        return episodic, semantic

    @classmethod
    def seed(cls, episodic: EpisodicMemory, distiller: KnowledgeDistiller,
             economy=None, bus=None, scorer=None) -> "LibrarianAgent":
        global _LIBRARIAN_SINGLETON
        seeded = os.environ.get("LIBRARIAN_SEEDED", "true").lower() != "false"
        agent  = cls(episodic, distiller, economy=economy, bus=bus,
                     scorer=scorer, seeded=seeded)
        _LIBRARIAN_SINGLETON = agent

        if economy is not None and seeded:
            economy.balances[cls.AGENT_ID] = 100

        return agent
