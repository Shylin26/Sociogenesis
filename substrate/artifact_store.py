import os
import json
import sqlite3
import hashlib
import time
from dataclasses import dataclass,field
from enum import Enum
from typing import Optional
from pathlib import Path

class ArtifactType(Enum):
    CODE="code"
    RESEARCH="research"
    VISUAL="visual"


@dataclass
class Artifact:
    artifact_id:str
    artifact_type:ArtifactType
    content:str
    author_id:int
    task_id:str
    tick:int
    coalition_id:Optional[str]=None
    quality_score:float=0.0
    timestamp:float=field(default_factory=time.monotonic)
    file_path:str =""
    links:list=field(default_factory=list)

    def to_dict(self)->dict:
        return{
            "artifact_id":self.artifact_id,
            "artifact_type":self.artifact_type.value,
            "author_id":self.author_id,
            "task_id":self.task_id,
            "coalition_id":self.coalition_id,
            "quality_score":round(self.quality_score,3),
            "tick":self.tick,
            "timestamp":round(self.timestamp,4),
            "file_path":self.file_path,
            "links":self.links,
        }
class ArtifactStore:
    _EXT={
        ArtifactType.CODE:".py",
        ArtifactType.RESEARCH:".md",
        ArtifactType.VISUAL:".txt",
    }
    def __init__(self,base_dir:str="./artifacts"):
        self.base_dir=Path(base_dir)
        self._setup_dirs()
        self._db_path=self.base_dir/"artifacts.db"
        self._conn=self._setup_db()

        self._cache:dict[str,Artifact]={}
        self.total_saved:int=0
        self.total_fetched:int=0


    def save(self,artifact_type:ArtifactType,content:str,author_id:int,task_id:str,tick:int,coalition_id:Optional[str]=None,quality_score:float=0.0,links:list=None)->Artifact:
        if links is None:
            links=[]
        artifact_id=self._content_hash(content)
        existing=self._cache.get(artifact_id)
        if existing:
            return existing
        
        file_path=self._write_file(artifact_type,artifact_id,content)
        artifact=Artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content=content,
            author_id=author_id,
            task_id=task_id,
            tick=tick,
            coalition_id=coalition_id,
            quality_score=quality_score,
            timestamp     = time.monotonic(),
            file_path     = str(file_path),
            links         = links,
        )
        self._db_insert(artifact)
        self._cache[artifact_id]=artifact
        self.total_saved+=1
        return artifact
    
    def update_quality(self,artifact_id:str,score:float):
        score=max(0.0,min(1.0,score))
        self._conn.execute(
            "UPDATE artifacts SET quality_score=? WHERE artifact_id=?",
            (score,artifact_id)
        )
        self._conn.commit()
        if artifact_id in self._cache:
            self._cache[artifact_id].quality_score=score
    
    def add_link(self,from_id:str,to_id:str):
        artifact=self._cache.get(from_id) or self.fetch(from_id)
        if artifact and to_id not in artifact.links:
            artifact.links.append(to_id)
            self._conn.execute(
                "UPDATE artifacts SET links=? WHERE artifact_id=?",
                (json.dumps(artifact.links),from_id)
            )
            self._conn.commit()
    
    def fetch(self,artifact_id:str)->Optional[Artifact]:
        if artifact_id in self._cache:
            self.total_fetched+=1
            return self._cache[artifact_id]
        
        row=self._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id=?",
            (artifact_id,)
        ).fetchone()
        if not row:
            return None
        
        artifact=self._row_to_artifact(row)
        self._cache[artifact_id]=artifact
        self.total_fetched+=1
        return artifact
    
    def fetch_content(self,artifact_id:str)->Optional[str]:
        artifact=self.fetch(artifact_id)
        if not artifact or not artifact.file_path:
            return None
        try:
            return Path(artifact.file_path).read_text()
        except FileNotFoundError:
            return None
    
    def by_agent(self,agent_id:int,limit:int=20)->list[Artifact]:
        rows=self._conn.execute(
            """SELECT * FROM artifacts
               WHERE author_id=?
               ORDER BY tick DESC LIMIT ?""",
            (agent_id,limit)
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]
    
    def by_type(self, artifact_type: ArtifactType,
                limit: int = 20) -> list[Artifact]:
        """All artifacts of a specific type, newest first."""
        rows = self._conn.execute(
            """SELECT * FROM artifacts
               WHERE artifact_type=?
               ORDER BY tick DESC LIMIT ?""",
            (artifact_type.value, limit)
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]
    
    def top_quality(self, n: int = 10,
                    artifact_type: Optional[ArtifactType] = None
                    ) -> list[Artifact]:
        """
        Top n artifacts by quality score.
        Optionally filtered by type.
        Used by HistorianAgent (Week 8) for the results section.
        """
        if artifact_type:
            rows = self._conn.execute(
                """SELECT * FROM artifacts WHERE artifact_type=?
                   ORDER BY quality_score DESC LIMIT ?""",
                (artifact_type.value, n)
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM artifacts
                   ORDER BY quality_score DESC LIMIT ?""",
                (n,)
            ).fetchall()
        return [self._row_to_artifact(r) for r in rows]
    
    def recent(self,n:int=5)->list[dict]:
        rows=self._conn.execute(
            "SELECT * FROM artifacts ORDER BY tick DESC LIMIT ?", (n,)

        ).fetchall()
        return [self._row_to_artifact(r).to_dict() for r in rows]
    
    def snapshot(self)->dict:
        return{
            "total_saved":self.total_saved,
            "total_fetched":self.total_fetched,
            "by_type":self.count(),
            "cache_size":len(self._cache),
        }
    
    def count(self)->dict:
        rows=self._conn.execute("SELECT artifact_type, COUNT(*) as c FROM artifacts GROUP BY artifact_type").fetchall()
        return {r["artifact_type"]: r["c"] for r in rows}
    
    def _setup_dirs(self):
        """Create a directory tree if it doesn't exist"""
        for t in ArtifactType:
            (self.base_dir / t.value).mkdir(parents=True, exist_ok=True)

    
    def _setup_db(self)->sqlite3.Connection:
        """Create SQLite DB and artifacts table."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id   TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                author_id     INTEGER NOT NULL,
                task_id       TEXT NOT NULL,
                coalition_id  TEXT,
                quality_score REAL DEFAULT 0.0,
                tick          INTEGER NOT NULL,
                timestamp     REAL NOT NULL,
                file_path     TEXT NOT NULL,
                links         TEXT DEFAULT '[]'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_author "
            "ON artifacts(author_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_type "
            "ON artifacts(artifact_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_quality "
            "ON artifacts(quality_score DESC)"
        )
        conn.commit()
        return conn
    
    def _write_file(self,artifact_type:ArtifactType,artifact_id:str,content:str)->Path:
        ext=self._EXT[artifact_type]
        filename=artifact_id[:16]+ext
        path     = self.base_dir / artifact_type.value / filename
        path.write_text(content, encoding="utf-8")
        return path
    
    def _db_insert(self,a:Artifact):
        self._conn.execute(
            """INSERT OR IGNORE INTO artifacts
               (artifact_id, artifact_type, author_id, task_id,
                coalition_id, quality_score, tick, timestamp,
                file_path, links)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                a.artifact_id,
                a.artifact_type.value,
                a.author_id,
                a.task_id,
                a.coalition_id,
                a.quality_score,
                a.tick,
                a.timestamp,
                a.file_path,
                json.dumps(a.links),
            )
        )
        self._conn.commit()

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        """Reconstruct an Artifact from a SQLite row (no content)."""
        return Artifact(
            artifact_id   = row["artifact_id"],
            artifact_type = ArtifactType(row["artifact_type"]),
            content       = "",        # load from disk only when needed
            author_id     = row["author_id"],
            task_id       = row["task_id"],
            tick          = row["tick"],
            coalition_id  = row["coalition_id"],
            quality_score = row["quality_score"],
            timestamp     = row["timestamp"],
            file_path     = row["file_path"],
            links         = json.loads(row["links"]),
        )
    
    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
    
    def close(self):
        self._conn.close()



if __name__=="__main__":
    import shutil

    TEST_DIR="/tmp/test_artifacts"
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    
    print("=== ArtifactStore smoke test ===\n")
    store = ArtifactStore(base_dir=TEST_DIR)

    a1 = store.save(
        ArtifactType.CODE,
        content="def binary_search(arr, x):\n    lo,hi=0,len(arr)-1\n    while lo<=hi:\n        mid=(lo+hi)//2\n        if arr[mid]==x: return mid\n        elif arr[mid]<x: lo=mid+1\n        else: hi=mid-1\n    return -1",
        author_id=3,
        task_id="task-001",
        tick=42,
        quality_score=0.88,
    )
    print(f"saved code artifact  id={a1.artifact_id[:12]}...")
    assert os.path.exists(a1.file_path), "file not written to disk"
    print(f"  file on disk: {a1.file_path}")

    hypothesis=json.dumps({
         "claim"        : "agents with higher task diversity earn more tokens",
        "evidence"     : ["token_balance_logs", "task_type_history"],
        "experiment"   : "compare token growth for mono vs diverse task agents",
        "falsifiable"  : True,

    })
    a2 = store.save(
        ArtifactType.RESEARCH,
        content=hypothesis,
        author_id=6,
        task_id="task-002",
        tick=55,
        quality_score=0.72,
    )
    print(f"saved research artifact  id={a2.artifact_id[:12]}...")
    a3 = store.save(
        ArtifactType.VISUAL,
        content="t-SNE plot of agent fingerprints at tick 55: 3 clusters visible",
        author_id=1,
        task_id="task-003",
        tick=55,
        quality_score=0.65,
    )
    print(f"saved visual artifact  id={a3.artifact_id[:12]}...")
    a1_dup = store.save(
        ArtifactType.CODE,
        content=a1.content,
        author_id=3,
        task_id="task-004",
        tick=60,
    )
    assert a1_dup.artifact_id == a1.artifact_id, "Dedup failed"
    print(f"deduplication works — same content → same artifact_id")

    fetched = store.fetch(a2.artifact_id)
    assert fetched.author_id == 6
    print(f"fetch by id works")

    content = store.fetch_content(a1.artifact_id)
    assert "binary_search" in content
    print(f"fetch_content reads from disk")

    store.update_quality(a3.artifact_id, 0.91)
    updated = store.fetch(a3.artifact_id)
    assert updated.quality_score == 0.91
    print(f"update_quality works")
 
    store.add_link(a2.artifact_id, a1.artifact_id)
    linked = store.fetch(a2.artifact_id)
    assert a1.artifact_id in linked.links
    print(f"cross-pollination link recorded")

    by_agent = store.by_agent(3)
    assert len(by_agent) >= 1
    top = store.top_quality(n=3)
    assert top[0].quality_score >= top[1].quality_score
    print(f" by_agent and top_quality queries work")
 
    
    snap = store.snapshot()
    print(f"\nSnapshot : {snap}")
    print(f"Recent   : {[r['artifact_type'] for r in store.recent(5)]}")
    print(f"Counts   : {store.count()}")
 
    store.close()
    shutil.rmtree(TEST_DIR)
    print("\n=== all assertions passed ===")




    





    




            



