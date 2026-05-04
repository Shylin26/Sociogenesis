"""
Marketplace component for the Sociogenesis system.
Provides a platform for agents to exchange resources and services.
"""

from typing import Dict, Any, List
import asyncio

class Marketplace:
    def __init__(self):
        self.listings: List[Dict[str, Any]] = []
        self.transactions: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def create_listing(self, agent_id: str, item: str, price: float) -> str:
        async with self._lock:
            listing_id = f"listing_{len(self.listings)}"
            listing = {
                "id": listing_id,
                "seller": agent_id,
                "item": item,
                "price": price,
                "status": "active"
            }
            self.listings.append(listing)
            return listing_id

    async def get_active_listings(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [l for l in self.listings if l["status"] == "active"]

    async def purchase(self, buyer_id: str, listing_id: str) -> bool:
        async with self._lock:
            for listing in self.listings:
                if listing["id"] == listing_id and listing["status"] == "active":
                    listing["status"] = "sold"
                    self.transactions.append({
                        "buyer": buyer_id,
                        "seller": listing["seller"],
                        "item": listing["item"],
                        "price": listing["price"]
                    })
                    return True
            return False
