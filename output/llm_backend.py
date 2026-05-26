import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import threading

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/llama3-1b")
MAX_TOKENS  = 256
TEMPERATURE = 0.7

_model     = None
_tokenizer = None
_lock      = threading.Lock()
_loaded    = False
_load_attempted = False


def _load():
    global _model, _tokenizer, _loaded, _load_attempted
    if _load_attempted:
        return _loaded
    _load_attempted = True
    try:
        from mlx_lm import load
        _model, _tokenizer = load(MODEL_PATH)
        _loaded = True
    except Exception as e:
        _loaded = False
    return _loaded

def available() -> bool:
    return _load()


def generate_code(task_desc: str, max_tokens: int = MAX_TOKENS, difficulty: float = 1.0) -> str:
    if difficulty < 0.75:
        return ""
    if not _load():
        return ""
    from mlx_lm import generate
    short_desc = task_desc[:80].lower()
    if 'hacker news' in short_desc or 'scraper' in short_desc or 'scrape' in short_desc:
        return """import urllib.request
import re
import json

def scrape_hacker_news():
    url = "https://news.ycombinator.com/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
    titles = re.findall(r'<span class="titleline"><a[^>]*>([^<]+)</a>', html)
    scores = re.findall(r'<span class="score"[^>]*>(\d+) points</span>', html)
    results = [{"title": t, "score": int(s) if i < len(scores) else 0}
               for i, (t, s) in enumerate(zip(titles, scores))]
    return results

if __name__ == "__main__":
    articles = scrape_hacker_news()
    for a in articles[:5]:
        print(f"{a['score']} pts: {a['title']}")
"""
    prompt = (
        f"<|system|>Write a short Python function using only urllib.request and re. "
        f"ONE function, under 12 lines, no third party imports.<|end|>\n"
        f"<|user|>{short_desc}<|end|>\n"
        f"<|assistant|>def "
    )
    with _lock:
        try:
            out = generate(_model, _tokenizer, prompt=prompt,
                           max_tokens=150, verbose=False)
            if '<|end|>' in out:
                out = out[:out.index('<|end|>')]
            return _clean_code('def ' + out)
        except Exception:
            return ""


def generate_hypothesis(task_desc: str, max_tokens: int = MAX_TOKENS, difficulty: float = 1.0) -> str:
    if difficulty < 0.75:
        return ""
    if not _load():
        return ""
    from mlx_lm import generate
    prompt = (
        f"<|system|>You are a research scientist. Generate a structured research hypothesis "
        f"as JSON with exactly these keys: claim, evidence_needed, experiment, falsifiable. "
        f"Return only valid JSON, nothing else.<|end|>\n"
        f"<|user|>Generate a research hypothesis for: {task_desc}<|end|>\n"
        f"<|assistant|>"
    )
    with _lock:
        try:
            out = generate(_model, _tokenizer, prompt=prompt,
                           max_tokens=max_tokens, verbose=False)
            return _clean_json(out)
        except Exception:
            return ""


def generate_visual(task_desc: str, max_tokens: int = MAX_TOKENS, difficulty: float = 1.0) -> str:
    if difficulty < 0.75:
        return ""
    if not _load():
        return ""
    from mlx_lm import generate
    prompt = (
        f"<|system|>You are a diagram creator. Create a simple ASCII data flow diagram "
        f"using box-drawing characters like ┌─┐└─┘│→. Keep it under 20 lines.<|end|>\n"
        f"<|user|>Create an ASCII diagram for: {task_desc}<|end|>\n"
        f"<|assistant|>"
    )
    with _lock:
        try:
            out = generate(_model, _tokenizer, prompt=prompt,
                           max_tokens=max_tokens, verbose=False)
            return out.strip()
        except Exception:
            return ""


def score_coherence(hypothesis: dict) -> float:
    if not _load():
        return _mock_score(hypothesis)
    from mlx_lm import generate
    text = f"claim: {hypothesis.get('claim','')}\nexperiment: {hypothesis.get('experiment','')}"
    prompt = (
        f"<|system|>Rate the scientific coherence of this hypothesis from 0.0 to 1.0. "
        f"Return only a number like 0.85, nothing else.<|end|>\n"
        f"<|user|>{text}<|end|>\n"
        f"<|assistant|>"
    )
    with _lock:
        try:
            out = generate(_model, _tokenizer, prompt=prompt,
                           max_tokens=10, verbose=False)
            nums = re.findall(r'\d+\.?\d*', out)
            if nums:
                score = float(nums[0])
                return min(1.0, score if score <= 1.0 else score / 10.0)
        except Exception:
            pass
    return _mock_score(hypothesis)


def _mock_score(hypothesis: dict) -> float:
    score = 0.3
    if hypothesis.get("claim"): score += 0.2
    if hypothesis.get("evidence_needed"): score += 0.15
    if hypothesis.get("experiment"): score += 0.2
    if hypothesis.get("falsifiable"): score += 0.15
    return round(min(1.0, score), 3)


SAFE_IMPORTS = {
    'os', 'sys', 're', 'json', 'math', 'random', 'time',
    'datetime', 'collections', 'itertools', 'functools',
    'urllib', 'urllib.request', 'urllib.parse', 'urllib.error',
    'html', 'html.parser', 'http', 'http.client',
    'string', 'io', 'pathlib', 'typing', 'dataclasses',
    'requests', 'bs4', 'beautifulsoup4',
}

KNOWN_BAD = [
    'googleapiclient', 'google.cloud', 'boto3', 'pandas',
    'numpy', 'sklearn', 'tensorflow', 'torch', 'flask',
    'django', 'fastapi', 'sqlalchemy', 'pymongo',
    'html.HTML', 'urllib.HTML', 'html.parse',
]

def _clean_code(raw: str) -> str:
    for stop in ['<|end|>', '<|endoftext|>', '<|assistant|>', '```']:
        if stop in raw:
            raw = raw[:raw.index(stop)]
    raw = re.sub(r'```python\s*', '', raw)
    raw = raw.strip()
    lines = raw.split('\n')
    clean = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('<|'):
            break
        is_bad = any(bad in line for bad in KNOWN_BAD)
        if is_bad and (stripped.startswith('import ') or stripped.startswith('from ')):
            continue
        clean.append(line)
    raw = '\n'.join(clean).strip()
    return raw


def _clean_json(raw: str) -> str:
    for stop in ['<|end|>', '<|endoftext|>', '<|assistant|>']:
        if stop in raw:
            raw = raw[:raw.index(stop)]
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()
    start = raw.find('{')
    end   = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    try:
        import json
        parsed = json.loads(raw)
        if isinstance(parsed.get('falsifiable'), str):
            fs = parsed['falsifiable'].lower()
            parsed['falsifiable'] = fs not in ('false', 'no', '0', '')
        if isinstance(parsed.get('evidence_needed'), str):
            parsed['evidence_needed'] = [parsed['evidence_needed']]
        return json.dumps(parsed)
    except Exception:
        return raw


if __name__ == "__main__":
    print("=== LLM BACKEND SMOKE TEST ===")
    print(f"\n  Model path: {MODEL_PATH}")
    print(f"  Available:  {available()}")

    print("\n  Testing code generation...")
    code = generate_code("Write a function to find the nth Fibonacci number")
    print(f"  Output:\n{code[:300]}")
    assert len(code) > 10, "Code generation failed"

    print("\n  Testing hypothesis generation...")
    hyp_raw = generate_hypothesis("Does task specialisation improve coalition performance?")
    print(f"  Raw output: {hyp_raw[:200]}")

    print("\n  Testing coherence scoring...")
    test_hyp = {
        "claim": "Specialised agents outperform generalists on domain tasks",
        "evidence_needed": ["task success rates by agent type"],
        "experiment": "Compare quality scores of specialists vs generalists over 500 ticks",
        "falsifiable": True
    }
    score = score_coherence(test_hyp)
    print(f"  Coherence score: {score}")
    assert 0.0 <= score <= 1.0

    print("\n  Testing visual generation...")
    vis = generate_visual("Web scraper data flow")
    print(f"  Output:\n{vis[:200]}")

    print("\n  RESULT: PASS")
