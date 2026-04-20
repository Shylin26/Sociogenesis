class SharedMemory:
    def __init__(self):
        self._memory = {}

    def read(self, key: str):
        return self._memory.get(key, None)

    def write(self, key: str, value: any):
        self._memory[key] = value

    def clear(self, key: str):
        if key in self._memory:
            del self._memory[key]

    def get_all_keys(self):
        return list(self._memory.keys())
