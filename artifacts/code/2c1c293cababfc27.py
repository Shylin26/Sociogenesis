
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
