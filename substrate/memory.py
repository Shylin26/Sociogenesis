import faiss
import numpy as np
class SharedMemory:
    def __init__(self,dim:int=128):
        self.index=faiss.IndexFlatL2(dim)
        # Maps the FAISS internal integer id to our metadata payload
        self.store={}
    def write(self,vec:np.ndarray,meta:dict):
        idx=self.index.ntotal
        self.index.add(vec.reshape(1,-1))
        self.store[idx]=meta
    def query(self,vec:np.ndarray,k:int=5)->dict:
        """Retrieve the metadata for the k nearest neighbours to the query vector."""
        if self.index.ntotal==0:
            return[]
        #Search returns D ,I 
        D,I=self.index.search(vec.reshape(1,-1),k)
        return [self.store[i] for i in I[0] if i in self.store]