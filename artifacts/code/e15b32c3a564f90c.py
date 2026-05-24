
def scrape_hackernews(html_content):
    """Parse HN front page HTML and return titles."""
    import re
    pattern = r'<span class="titleline"><a[^>]*>([^<]+)</a>'
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
