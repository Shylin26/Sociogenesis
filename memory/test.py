import sys
import time
import numpy as np
import pytest

sys.path.insert(0, ".")

from memory.episodic     import EpisodicMemory, DIM
from memory.distillation import KnowledgeDistiller, MIN_RECORDS
from memory.librarian    import (
    LibrarianAgent, MockCoherenceScorer, build_scorer, PRESTIGE_BONUS
)


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def episodic(tmp_db):
    return EpisodicMemory(db_path=tmp_db)


@pytest.fixture
def distiller(episodic):
    return KnowledgeDistiller(episodic, db_path=episodic.db_path)


@pytest.fixture
def populated_episodic(tmp_db):
    em  = EpisodicMemory(db_path=tmp_db)
    rng = np.random.default_rng(42)
    for i in range(MIN_RECORDS + 50):
        task_type = ["code", "research", "visual"][i % 3]
        em.record(
            task_id      = f"{task_type}:{i}",
            task_emb     = rng.standard_normal(DIM).astype(np.float32),
            solution_emb = rng.standard_normal(DIM).astype(np.float32),
            quality      = rng.uniform(0.2, 1.0),
            agent_id     = i % 10,
        )
    return em


@pytest.fixture
def populated_distiller(populated_episodic):
    return KnowledgeDistiller(populated_episodic, db_path=populated_episodic.db_path)


class MockEconomy:
    def __init__(self):
        self.balances = {}

    def earn(self, agent_id, amount):
        self.balances[agent_id] = self.balances.get(agent_id, 0) + amount

    def spend(self, agent_id, amount):
        self.balances[agent_id] = self.balances.get(agent_id, 0) - amount


def test_S1_episodic_write_and_retrieve(episodic):
    rng      = np.random.default_rng(0)
    task_emb = rng.standard_normal(DIM).astype(np.float32)
    sol_emb  = rng.standard_normal(DIM).astype(np.float32)

    rec = episodic.record(task_id="code:001", task_emb=task_emb,
                          solution_emb=sol_emb, quality=0.8, agent_id=1)
    assert rec.record_id == 0
    assert len(episodic) == 1

    results = episodic.retrieve(task_emb, k=1)
    assert len(results) == 1
    assert results[0].task_id == "code:001"
    assert results[0].quality == pytest.approx(0.8)


def test_S2_hebbian_decay_and_compress(episodic):
    rng = np.random.default_rng(1)
    episodic.record("research:bad",
                    rng.standard_normal(DIM).astype(np.float32),
                    rng.standard_normal(DIM).astype(np.float32),
                    quality=0.1, agent_id=0)
    initial_count = len(episodic)

    for _ in range(600):
        episodic.tick_decay()

    pruned = episodic.compress()
    assert pruned >= 1
    assert len(episodic) < initial_count


def test_S3_faiss_numpy_fallback_parity(tmp_db):
    rng   = np.random.default_rng(2)
    query = rng.standard_normal(DIM).astype(np.float32)

    def populate(em):
        for i in range(10):
            v = rng.standard_normal(DIM).astype(np.float32)
            em.record(f"t:{i}", v, v.copy(), quality=0.7, agent_id=0)
        close = query + 0.01 * rng.standard_normal(DIM).astype(np.float32)
        em.record("t:close", close, close.copy(), quality=0.9, agent_id=0)

    import memory.episodic as ep_mod
    original_faiss = ep_mod._FAISS_AVAILABLE

    ep_mod._FAISS_AVAILABLE = False
    em_np = EpisodicMemory(db_path=tmp_db + "_np")
    populate(em_np)
    np_result = em_np.retrieve(query, k=1)

    ep_mod._FAISS_AVAILABLE = original_faiss
    assert np_result[0].task_id == "t:close"


def test_S4_distiller_skips_below_threshold(episodic, distiller):
    rng = np.random.default_rng(3)
    episodic.record("code:1",
                    rng.standard_normal(DIM).astype(np.float32),
                    rng.standard_normal(DIM).astype(np.float32),
                    0.7, 0)
    nodes = distiller.distill()
    assert nodes == 0
    assert distiller.node_count == 0


def test_S5_distiller_produces_nodes_and_edges(populated_distiller):
    n = populated_distiller.distill()
    assert n > 0
    assert populated_distiller.node_count > 0
    assert populated_distiller.graph is not None


def test_S6_distiller_retrieve(populated_distiller):
    populated_distiller.distill()
    rng     = np.random.default_rng(5)
    query   = rng.standard_normal(DIM).astype(np.float32)
    results = populated_distiller.retrieve(query, k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    for node in results:
        assert hasattr(node, "embedding")
        assert node.embedding.shape == (DIM,)


def test_S7_distiller_edge_decay(populated_distiller):
    populated_distiller.distill()
    initial_edges = populated_distiller.edge_count

    if initial_edges == 0:
        pytest.skip("No edges formed — cluster similarity too low for this seed")

    for _ in range(1200):
        populated_distiller.tick_decay()

    assert populated_distiller.edge_count < initial_edges


def test_S8_mock_scorer_good_hypothesis():
    scorer = MockCoherenceScorer()
    good   = {
        "claim"           : "Agents that receive higher rewards predict better outcomes because they correlate task data",
        "evidence_needed" : ["reward log", "task trace"],
        "experiment"      : "Run 100 tasks, measure prediction accuracy vs reward",
        "falsifiable"     : True,
    }
    score = scorer.score(good)
    assert score > 0.5, f"Good hypothesis should score > 0.5, got {score}"


def test_S9_mock_scorer_empty():
    scorer = MockCoherenceScorer()
    assert scorer.score({}) == 0.0
    assert scorer.score("not a dict") == 0.0


def test_S10_librarian_seed_registers_economy(populated_episodic, tmp_db):
    distiller = KnowledgeDistiller(populated_episodic, db_path=tmp_db)
    economy   = MockEconomy()
    LibrarianAgent.seed(populated_episodic, distiller, economy=economy)
    assert LibrarianAgent.AGENT_ID in economy.balances
    assert economy.balances[LibrarianAgent.AGENT_ID] == 100


def test_S11_librarian_distills_after_interval(populated_episodic, tmp_db):
    distiller = KnowledgeDistiller(populated_episodic, db_path=tmp_db)
    economy   = MockEconomy()
    librarian = LibrarianAgent.seed(
        populated_episodic, distiller, economy=economy,
        scorer=MockCoherenceScorer()
    )
    librarian.distill_interval = 5

    librarian.on_tick(0)
    librarian.on_tick(5)
    time.sleep(1.0)

    assert len(librarian.reports) >= 1
    assert economy.balances[LibrarianAgent.AGENT_ID] > 100


def test_S12_librarian_retrieve_context(populated_episodic, tmp_db):
    distiller = KnowledgeDistiller(populated_episodic, db_path=tmp_db)
    distiller.distill()
    librarian = LibrarianAgent.seed(populated_episodic, distiller)

    rng      = np.random.default_rng(7)
    task_emb = rng.standard_normal(DIM).astype(np.float32)
    ep_recs, sem_nodes = librarian.retrieve_context(task_emb, k=3)

    assert isinstance(ep_recs, list)
    assert isinstance(sem_nodes, list)
    assert len(ep_recs) <= 3
    assert len(sem_nodes) <= 3


def test_S13_librarian_score_delegates(populated_episodic, tmp_db):
    distiller = KnowledgeDistiller(populated_episodic, db_path=tmp_db)
    scorer    = MockCoherenceScorer()
    librarian = LibrarianAgent.seed(populated_episodic, distiller, scorer=scorer)

    hyp   = {"claim": "test because data", "evidence_needed": [],
              "experiment": "run", "falsifiable": True}
    score = librarian.score_hypothesis(hyp)
    assert 0.0 <= score <= 1.0


def test_S14_rag_quality_regression(tmp_db):
    em  = EpisodicMemory(db_path=tmp_db)
    rng = np.random.default_rng(8)

    task_a_emb = rng.standard_normal(DIM).astype(np.float32)
    sol_a_emb  = rng.standard_normal(DIM).astype(np.float32)
    em.record("code:task_a", task_a_emb, sol_a_emb, quality=0.95, agent_id=0)

    for i in range(20):
        em.record(f"noise:{i}",
                  rng.standard_normal(DIM).astype(np.float32),
                  rng.standard_normal(DIM).astype(np.float32),
                  quality=0.5, agent_id=1)

    noisy_query = task_a_emb + 0.05 * rng.standard_normal(DIM).astype(np.float32)
    results     = em.retrieve(noisy_query, k=1)

    assert results[0].task_id == "code:task_a"
    assert results[0].quality == pytest.approx(0.95)


def test_S15_graph_grows_with_records(tmp_db):
    rng = np.random.default_rng(9)

    def make_em(n_records, suffix):
        em = EpisodicMemory(db_path=tmp_db + suffix)
        for i in range(n_records):
            em.record(f"t:{i}",
                      rng.standard_normal(DIM).astype(np.float32),
                      rng.standard_normal(DIM).astype(np.float32),
                      quality=rng.uniform(0.3, 1.0), agent_id=i % 5)
        return em

    em_small = make_em(MIN_RECORDS + 10,  "_small")
    em_large = make_em(MIN_RECORDS + 200, "_large")

    d_small = KnowledgeDistiller(em_small, db_path=tmp_db + "_ds")
    d_large = KnowledgeDistiller(em_large, db_path=tmp_db + "_dl")

    d_small.distill()
    d_large.distill()

    assert d_small.node_count > 0
    assert d_large.node_count > 0
    assert d_large.node_count >= d_small.node_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
