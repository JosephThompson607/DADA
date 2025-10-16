import heapq
from dataclasses import dataclass, field
import networkx as nx
from set_new_edges import *

@dataclass(order=True)
class Solution:
    objective: float
    edges: set = field(compare=False, default_factory=set)

class EliteSet:
    """Keeps the best (lowest objective) solutions."""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._heap = []  # min-heap by objective (smallest = best)

    def add(self, solution: Solution):
        # If there’s room, add directly
        
        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, (-solution.objective, solution))  # store as negative to invert
        else:
            # Replace the *worst* (largest objective) if new one is smaller
            #Doing <= so we get the full depth of edges, even in a tie.
            if solution.objective <= -self._heap[0][0]:
                heapq.heapreplace(self._heap, (-solution.objective, solution))

    def get_elites(self, sorted=False):
        """Return solutions sorted by objective ascending (best first)."""
        sols = [s for _, s in self._heap]
        if sorted:
            return sorted(sols, key=lambda s: s.objective)
        else:
            return sols


    def best(self):
        """Return the best (lowest objective) solution."""
        return heapq.nlargest(1,self._heap, key=lambda x: x[0])[0][1]

    def worst(self):
        """Return the worst (highest objective) solution."""
        return self._heap[0][1]

    def is_empty(self):
        return len(self._heap) == 0
    



def remove_edges(G_max_close_orig, removed_edges):
    G_max_close= G_max_close_orig.copy()
    G_max_close.remove_edges_from(removed_edges)
    return nx.transitive_reduction(G_max_close)

def beam_search_mh( orig_salbp, G_max_close_orig,G_min, mh, width=1, depth=1 , **mhkwargs):
    
    G_max_close= G_max_close_orig.copy()
    G_max_red = nx.transitive_reduction(G_max_close)
    history = set()
    elites = EliteSet(width)
    queue = [(None, [])]  # Queue of (obj_val, removed_edges) tuples
    for _ in range(depth):
        while len(queue) > 0:
            obj_val, removed_edges = queue.pop(0)
            edges = get_possible_edges(G_max_red, G_min)
            for i, edge in enumerate(edges):
                        #This is to avoid repeat computations (i.e. e1->e2, e2->e1)
                new_removed = removed_edges+ [edge]
                edge_set = frozenset(new_removed)
                if edge_set in history:
                    continue
                # Store the current removal sequence
                history.add(edge_set)
                # Create list without current item
                G_max_red = remove_edges(G_max_close, new_removed)
                new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
                res = mh(new_salbp, **mhkwargs)
                sol = Solution(res['n_stations'], new_removed)
                elites.add(sol)
        queue = elites.get_elites()
    
    return elites.best()
