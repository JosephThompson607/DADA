import heapq
from dataclasses import dataclass, field
import networkx as nx
from set_new_edges import *
from ml_search import *

@dataclass(order=True)
class Solution:
    objective: float
    edges: list = field(compare=False, default_factory=list)

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
    
    def same_first_edge(self) -> bool:
        """Return True if the first edge is the same for all elite solutions."""
        if len(self._heap) <= 1:
            # If 0 or 1 elite, trivially True
            return True

        elites = [s for _, s in self._heap]
        first_edges = [sol.edges[0] for sol in elites if sol.edges]  # ignore empty edges

        # If any solution has no edges, return False
        if len(first_edges) != len(elites):
            return False

        # Check if all first edges are identical
        return len(set(first_edges)) == 1



def remove_edges(G_max_close_orig, removed_edges):
    G_max_close= G_max_close_orig.copy()
    G_max_close.remove_edges_from(removed_edges)
    return nx.transitive_reduction(G_max_close)

def beam_search_mh( orig_salbp, G_max_close_orig,G_min, mh, beam_config = {"width":1, "depth":1} , **mhkwargs):
    width = beam_config["width"]
    depth = beam_config["depth"]
    G_max_close= G_max_close_orig.copy()
    G_max_red = nx.transitive_reduction(G_max_close)
    #Initial solution
    new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    res = mh(new_salbp, **mhkwargs)
    init_sol = Solution(res['n_stations'], [])
    history = set()
    elites = EliteSet(width)
    queue = [init_sol]  # Queue of Solution(obj_val, removed_edges) 
    for c_d in range(depth):
        while len(queue) > 0:
            old_sol= queue.pop(0)
            removed_edges = old_sol.edges
            edges = get_possible_edges(G_max_red, G_min)
            for i, edge in enumerate(edges):
                        #This is to avoid repeat computations (i.e. e1->e2, e2->e1)
                new_removed = removed_edges+ [(edge[0], edge[1])]
                edge_set = frozenset(new_removed)
                if edge_set in history:
                    continue
                # Store the current removal sequence
                history.add(edge_set)
                # Create list without current item
                G_max_red = remove_edges(G_max_close, new_removed)
                new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
                res = mh(new_salbp, **mhkwargs)
                if len(edge) ==3:
                    probability = edge[2]
                else:
                    probability = 1
                reward = max(0, probability*(old_sol.objective - res['n_stations']))
                sol = Solution(reward, new_removed)
                elites.add(sol)
        if c_d > 1:
            if elites.same_first_edge():
                
                print("elites agree on next move. proceeding to query")
                print(elites.get_elites())
                break
        queue = elites.get_elites()
    sol = elites.best()
    obj = sol.objective
    edges = sol.edges
    return edges[0], obj

#best_first_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model,**new_kwargs):
def beam_search_ml( orig_salbp, G_max_close_orig,G_min, ml_model, beam_config = {"width":1, "depth":1} , discount_factor = 1,**mhkwargs):
    width = beam_config["width"]
    depth = beam_config["depth"]
    G_max_close= G_max_close_orig.copy()
    G_max_red = nx.transitive_reduction(G_max_close)
    history = set() #Hash table of previous results
    elites = EliteSet(width)
    queue = [Solution(0, [])]  # Queue of Solution(obj_val, removed_edges) 
    for c_d in range(depth):
        while len(queue) > 0:
            old_sol= queue.pop(0)
            to_remove = old_sol.edges.copy()
            print("removing ", to_remove)
            G_max_red = remove_edges(G_max_close, to_remove)   
            edges = get_possible_edges(G_max_red, G_min)
            print("current edges", edges)
            edge_res = best_first_ml_choice_edge(edges,orig_salbp, G_max_red, ml_model, top_n=width, **mhkwargs)
            for edge, probability in edge_res:
                        #This is to avoid repeat computations (i.e. e1->e2, e2->e1)
                new_removed = to_remove+ [(edge[0], edge[1])]
                edge_set = frozenset(new_removed)
                if edge_set in history:
                    continue
                # Store the current removal sequence
                history.add(edge_set)
                #We do not have real objective value, here we are maximizing the sum of the probabilities
                #Subtracting probability so it becomes a minimization problem so heap datastructure works
                new_val = old_sol.objective - discount_factor*probability 
                sol = Solution(new_val, new_removed)
                elites.add(sol)
        queue = elites.get_elites()
        if c_d >= 1:
            if elites.same_first_edge():
                
                print("elites agree on next move. proceeding to query")
                print(elites.get_elites())
                break

    print("current elites ", elites.get_elites())
    sol = elites.best()
    print("best is ", sol)
    obj = sol.objective
    edges = sol.edges
    return edges[0], obj
