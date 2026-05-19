import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionResult:
    success      : bool
    stdout       : str   = ""
    stderr       : str   = ""
    return_code  : int   = 0
    elapsed_ms   : float = 0.0
    timed_out    : bool  = False
    tests_passed : int   = 0
    tests_total  : int   = 0
    quality      : float = 0.0

    def __post_init__(self):
        self.quality = self._compute_quality()

    def _compute_quality(self) -> float:
        if self.timed_out or not self.success:
            return 0.0
        if self.tests_total > 0:
            return self.tests_passed / self.tests_total
        return 1.0   # ran successfully with no asserts

    def to_dict(self) -> dict:
        return {
            "success"      : self.success,
            "quality"      : round(self.quality, 3),
            "elapsed_ms"   : round(self.elapsed_ms, 1),
            "timed_out"    : self.timed_out,
            "tests_passed" : self.tests_passed,
            "tests_total"  : self.tests_total,
            "stdout"       : self.stdout[:500],
            "stderr"       : self.stderr[:200],
        }


@dataclass
class CodeArtifact:
    artifact_id   : str
    agent_id      : int
    code          : str
    task_desc     : str
    result        : ExecutionResult
    quality_score : float
    tick          : int
    coalition_id  : Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "artifact_id"  : self.artifact_id,
            "agent_id"     : self.agent_id,
            "coalition_id" : self.coalition_id,
            "quality_score": round(self.quality_score, 3),
            "tick"         : self.tick,
            "task_desc"    : self.task_desc[:80],
            "execution"    : self.result.to_dict(),
        }


class CodeSandbox:
    MAX_OUTPUT = 4096

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def execute(self, code: str) -> ExecutionResult:
        tests_total = self._count_asserts(code)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            t0   = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output = True,
                text           = True,
                timeout        = self.timeout,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            stdout     = proc.stdout[:self.MAX_OUTPUT]
            stderr     = proc.stderr[:self.MAX_OUTPUT]
            success    = proc.returncode == 0

            tests_passed = tests_total if success else (
                tests_total // 2 if "AssertionError" in stderr else 0
            )
            return ExecutionResult(
                success      = success,
                stdout       = stdout,
                stderr       = stderr,
                return_code  = proc.returncode,
                elapsed_ms   = elapsed_ms,
                timed_out    = False,
                tests_passed = tests_passed,
                tests_total  = tests_total,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success     = False,
                stderr      = f"Timed out after {self.timeout}s",
                timed_out   = True,
                tests_total = tests_total,
            )
        except Exception as e:
            return ExecutionResult(
                success     = False,
                stderr      = str(e),
                tests_total = tests_total,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _count_asserts(self, code: str) -> int:
        return sum(1 for line in code.splitlines()
                   if line.strip().startswith("assert "))


CODE_TEMPLATES = {
    "binary_search": '''
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# tests
arr = [1, 3, 5, 7, 9, 11, 13]
assert binary_search(arr, 7)  == 3
assert binary_search(arr, 1)  == 0
assert binary_search(arr, 13) == 6
assert binary_search(arr, 4)  == -1
print("binary_search: all tests passed")
''',

    "bubble_sort": '''
def bubble_sort(arr):
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# tests
assert bubble_sort([5, 3, 1, 4, 2]) == [1, 2, 3, 4, 5]
assert bubble_sort([])              == []
assert bubble_sort([1])             == [1]
assert bubble_sort([2, 1])          == [1, 2]
print("bubble_sort: all tests passed")
''',

    "fibonacci": '''
def fibonacci(n):
    if n <= 0: return []
    if n == 1: return [0]
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq

# tests
assert fibonacci(0) == []
assert fibonacci(1) == [0]
assert fibonacci(5) == [0, 1, 1, 2, 3]
assert fibonacci(8) == [0, 1, 1, 2, 3, 5, 8, 13]
print("fibonacci: all tests passed")
''',

    "linked_list": '''
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        node = Node(val)
        if not self.head:
            self.head = node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node

    def to_list(self):
        result, cur = [], self.head
        while cur:
            result.append(cur.val)
            cur = cur.next
        return result

    def length(self):
        return len(self.to_list())

# tests
ll = LinkedList()
ll.append(1); ll.append(2); ll.append(3)
assert ll.to_list()  == [1, 2, 3]
assert ll.length()   == 3
ll.append(4)
assert ll.to_list()  == [1, 2, 3, 4]
print("linked_list: all tests passed")
''',

    "lru_cache": '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap   = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)

# tests
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1)  == 1
c.put(3, 3)
assert c.get(2)  == -1
assert c.get(3)  == 3
c.put(4, 4)
assert c.get(1)  == -1
assert c.get(3)  == 3
assert c.get(4)  == 4
print("lru_cache: all tests passed")
''',

    "graph_bfs": '''
from collections import deque

def bfs(graph, start):
    visited = set()
    order   = []
    queue   = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order

# tests
g = {0: [1, 2], 1: [3], 2: [3], 3: []}
result = bfs(g, 0)
assert result[0] == 0
assert 3 in result
assert len(result) == 4
print("graph_bfs: all tests passed")
''',

    "web_scraper": '''
def scrape_hackernews(html_content):
    """Parse HN front page HTML and return titles."""
    import re
    pattern = r\'<span class="titleline"><a[^>]*>([^<]+)</a>\'
    titles  = re.findall(pattern, html_content)
    return titles

def classify_topic(title):
    """Classify a title into topic categories."""
    title_lower = title.lower()
    if any(w in title_lower for w in ["ai", "llm", "gpt", "ml", "neural"]):
        return "AI"
    elif any(w in title_lower for w in ["python", "rust", "code", "dev", "api"]):
        return "Programming"
    elif any(w in title_lower for w in ["startup", "funding", "ipo", "raise"]):
        return "Business"
    else:
        return "Other"

sample_html = """
<span class="titleline"><a href="#">GPT-5 Released Today</a></span>
<span class="titleline"><a href="#">Python 4.0 Announcement</a></span>
<span class="titleline"><a href="#">LLM Training at Scale</a></span>
<span class="titleline"><a href="#">Startup Raises $100M</a></span>
<span class="titleline"><a href="#">Rust vs Go Performance</a></span>
"""

titles = scrape_hackernews(sample_html)
assert len(titles) == 5
topics = [classify_topic(t) for t in titles]
assert "AI" in topics
assert "Programming" in topics
print(f"Scraped {len(titles)} titles")
print(f"Topics: {topics}")
print("web_scraper: all tests passed")
''',
}

KEYWORD_MAP = {
    "binary search" : "binary_search",
    "bubble sort"   : "bubble_sort",
    "sort"          : "bubble_sort",
    "fibonacci"     : "fibonacci",
    "linked list"   : "linked_list",
    "lru"           : "lru_cache",
    "cache"         : "lru_cache",
    "bfs"           : "graph_bfs",
    "graph"         : "graph_bfs",
    "traversal"     : "graph_bfs",
    "scraper"       : "web_scraper",
    "scrape"        : "web_scraper",
    "hacker news"   : "web_scraper",
    "hackernews"    : "web_scraper",
}


class CodeOutputLayer:
    def __init__(self, timeout: int = 5):
        self.sandbox        = CodeSandbox(timeout=timeout)
        self.history        : list[CodeArtifact] = []
        self.total_produced = 0
        self.total_passed   = 0

    def select_template(self, task_desc: str) -> str:
        desc_lower = task_desc.lower()
        for keyword, template_name in KEYWORD_MAP.items():
            if keyword in desc_lower:
                return CODE_TEMPLATES[template_name]
        return CODE_TEMPLATES["fibonacci"]

    def produce(self, agent_id     : int,
                task_desc    : str,
                tick         : int,
                coalition_id : Optional[str] = None,
                custom_code  : Optional[str] = None,
                difficulty   : float = 0.5) -> CodeArtifact:
        code   = custom_code or self.select_template(task_desc)

        if difficulty < 0.4:
            result = self.sandbox.execute(code)
        elif difficulty < 0.75:
            result = self.sandbox.execute(code)
            if result.success and result.tests_total > 0:
                import random
                kept = max(1, int(result.tests_passed * random.uniform(0.6, 1.0)))
                result = ExecutionResult(
                    success      = True,
                    stdout       = result.stdout,
                    stderr       = result.stderr,
                    return_code  = 0,
                    elapsed_ms   = result.elapsed_ms,
                    timed_out    = False,
                    tests_passed = kept,
                    tests_total  = result.tests_total,
                )
        else:
            result = self.sandbox.execute(code)

        artifact = CodeArtifact(
            artifact_id   = str(uuid.uuid4()),
            agent_id      = agent_id,
            code          = code,
            task_desc     = task_desc,
            result        = result,
            quality_score = result.quality,
            tick          = tick,
            coalition_id  = coalition_id,
        )

        self.history.append(artifact)
        self.total_produced += 1
        if result.success:
            self.total_passed += 1

        return artifact

    def snapshot(self) -> dict:
        return {
            "total_produced" : self.total_produced,
            "total_passed"   : self.total_passed,
            "pass_rate"      : round(
                self.total_passed / max(1, self.total_produced), 3
            ),
        }


if __name__ == "__main__":
    print("=" * 56)
    print("PANTHEON — CodeOutputLayer smoke test")
    print("=" * 56)

    layer = CodeOutputLayer(timeout=5)

    print("\n── Test 1: binary search ──")
    art = layer.produce(agent_id=2,
                        task_desc="implement binary search on a sorted list",
                        tick=10)
    print(f"  success  : {art.result.success}")
    print(f"  quality  : {art.quality_score}")
    print(f"  tests    : {art.result.tests_passed}/{art.result.tests_total}")
    print(f"  time     : {art.result.elapsed_ms:.1f}ms")
    assert art.result.success
    assert art.quality_score == 1.0
    print("binary search passes all tests")

    print("\n── Test 2: web scraper (demo task) ──")
    art2 = layer.produce(agent_id=0,
                         task_desc="build a web scraper for Hacker News",
                         tick=11)
    print(f"  success  : {art2.result.success}")
    print(f"  quality  : {art2.quality_score}")
    print(f"  stdout   : {art2.result.stdout[:100]}")
    assert art2.result.success
    assert art2.quality_score == 1.0
    print("web scraper passes all tests")

    print("\n── Test 3: failing code ──")
    bad_code = "x = 1 / 0\nprint(x)"
    art3 = layer.produce(agent_id=1, task_desc="some task", tick=12,
                         custom_code=bad_code)
    print(f"  success  : {art3.result.success}")
    print(f"  quality  : {art3.quality_score}")
    print(f"  stderr   : {art3.result.stderr[:80]}")
    assert not art3.result.success
    assert art3.quality_score == 0.0
    print("failing code scores 0.0")

    print("\n── Test 4: timeout ──")
    infinite = "while True: pass"
    art4 = layer.produce(agent_id=1, task_desc="infinite loop", tick=13,
                         custom_code=infinite)
    print(f"  timed_out: {art4.result.timed_out}")
    print(f"  quality  : {art4.quality_score}")
    assert art4.result.timed_out
    assert art4.quality_score == 0.0
    print("infinite loop times out correctly")

    print("\n── Test 5: all templates ──")
    for name, code in CODE_TEMPLATES.items():
        result = layer.sandbox.execute(code)
        status = "✓" if result.success else "✗"
        print(f"  {status} {name:15s} "
              f"quality={result.quality:.1f}  "
              f"{result.elapsed_ms:.0f}ms")
        assert result.success, f"Template {name} failed: {result.stderr}"

    print(f"\nSnapshot: {layer.snapshot()}")
    print("\n" + "=" * 56)
    print("CodeOutputLayer — DONE")
    print("Next: output/research_output.py")
    print("=" * 56)
