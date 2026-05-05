"""
Knowledge Base component for the Sociogenesis system.
Provides a centralized repository for facts and generalized information discovered by agents.
"""

from typing import Dict, Any, List, Optional
import asyncio

class KnowledgeBase:
    def __init__(self):
        # Maps topic/entity to a list of recorded facts or data
        self.knowledge: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def add_fact(self, topic: str, fact: Any, source_agent: str) -> None:
        async with self._lock:
            if topic not in self.knowledge:
                self.knowledge[topic] = []
            
            self.knowledge[topic].append({
                "fact": fact,
                "source": source_agent,
                "timestamp": asyncio.get_event_loop().time()
            })

    async def get_facts(self, topic: str) -> List[Dict[str, Any]]:
        async with self._lock:
            return self.knowledge.get(topic, []).copy()

    async def search(self, keyword: str) -> Dict[str, List[Dict[str, Any]]]:
        async with self._lock:
            results = {}
            for topic, facts in self.knowledge.items():
                if keyword.lower() in topic.lower():
                    results[topic] = facts
            return results
