import heapq
from dataclasses import dataclass, field
import networkx as nx
from set_new_edges import *
from ml_search import *
import random

@dataclass(order=True)
class Solution:
    accumulated_reward: float  # Higher is better
    state_probability: float
    value: float
    query_cost: int = field(compare=False) #Do not want to compare
    remaining_budget: int = field(compare=False)
    edges: list = field(compare=False, default_factory=list)



class EliteSet:
    """Keeps the best (highest accumulated_reward, then highest state_probability) solutions."""
    def __init__(self, max_size: int, diversity_bonus=1e-6):
        self.max_size = max_size
        self._heap = []  # min-heap by (accumulated_reward, state_probability)
        self._first_edges = set()  # cache of first edges in the elite set
        self.diversity_bonus = diversity_bonus

    def _update_first_edges(self):
        """Recompute the set of first edges in the current heap."""
        self._first_edges = {
            s.edges[0] for _, s in self._heap if s.edges
        }


    def add(self, solution: Solution):
             # Apply diversity bonus if this first edge is *not* already in the elite set
        diversity_bonus = 0.0
        if solution.edges and solution.edges[0] not in self._first_edges:
            diversity_bonus = self.diversity_bonus

        # Key for heap: prioritize higher reward, then higher probability
        key = (solution.accumulated_reward + diversity_bonus, solution.state_probability + diversity_bonus)

        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, (key, solution))
            self._update_first_edges()
        else:
            # Compare with the *worst* in heap (smallest key)
            worst_key, _ = self._heap[0]

            # Replace if new solution is better:
            #   higher reward OR same reward but higher probability
            if key > worst_key:
                heapq.heapreplace(self._heap, (key, solution))
                self._update_first_edges()

    def get_elites(self, sort_elites=False):
        """Return solutions sorted by accumulated_reward, then probability descending (best first)."""
        sols = [s for _, s in self._heap]
        if sort_elites:
            return sorted(
                sols,
                key=lambda s: (s.accumulated_reward, s.state_probability),
                reverse=True
            )
        else:
            return sols

    def best(self):
        """Return the best (highest accumulated_reward, then highest state_probability) solution."""
        return max(self._heap, key=lambda x: (x[0][0], x[0][1]))[1]

    def worst(self):
        """Return the worst (lowest accumulated_reward, then lowest state_probability) solution."""
        return self._heap[0][1]

    def is_empty(self):
        return len(self._heap) == 0

    def same_first_edge(self) -> bool:
        """Return True if the first edge is the same for all elite solutions."""
        if len(self._heap) <= 1:
            return True  # Trivially True

        elites = [s for _, s in self._heap]
        first_edges = [sol.edges[0] for sol in elites if sol.edges]

        # If any solution has no edges, return False
        if len(first_edges) != len(elites):
            return False

        return len(set(first_edges)) == 1


def remove_edges(G_max_close_orig, removed_edges):
    G_max_close= G_max_close_orig.copy()
    G_max_close.remove_edges_from(removed_edges)
    G_max_red = nx.transitive_reduction(G_max_close)
    G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
    return G_max_red

def beam_search_mh( orig_salbp, G_max_close,G_min, mh, remaining_budget = 1e6, rng = None, beam_config = {"width":1, "depth":1} , init_sol = None,mode='beam_mh', **mhkwargs):
    width = beam_config["width"]
    depth = beam_config["depth"]
    G_max_red = nx.transitive_reduction(G_max_close)
    G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
    #Initial solution
    if mode == 'beam_mh':
        if not init_sol:
            res = mh(new_salbp, **mhkwargs)
        else:
            res = init_sol
    elif mode == 'beam_prob':
        res = {'n_stations':0}
        if not rng:
            rng = random.Random()
    init_sol = Solution(0,1,res['n_stations'],None,remaining_budget, []  )
    history = set()
    best_sol = init_sol
    queue = [init_sol]  # Queue of Solution(obj_val, removed_edges) 
    for c_d in range(depth):
        elites = EliteSet(width)

        while len(queue) > 0:
            old_sol= queue.pop(0)
            removed_edges = old_sol.edges
            query_cost = old_sol.query_cost
            edges = get_possible_edges(G_max_red, G_min,remaining_budget=remaining_budget, ban_list=removed_edges)
            for i, edge in enumerate(edges):
                        #This is to avoid repeat computations (i.e. e1->e2, e2->e1)
                edge_prob = edge[2]
                edge_time_cost = edge[3]
                #Edge management
                new_removed = removed_edges+ [(edge[0], edge[1])]
                edge_set = frozenset(new_removed)
                if edge_set in history:
                    continue
                # Store the current removal sequence
                history.add(edge_set)
  
                prob = old_sol.state_probability * edge_prob #Overall likelihood is the probability of reaching previous states times the probability of the current
                if mode == 'beam_mh':
                    # remove edge and re-solve
                    G_max_red = remove_edges(G_max_close, new_removed)
                    G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
                    new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
                    res = mh(new_salbp, **mhkwargs)
                    current_val = res['n_stations']
                    reward = old_sol.accumulated_reward+ max( 0, prob*(old_sol.value - current_val))/edge_time_cost
                elif mode == 'beam_prob':
                    #Value is 1 for removing the edge times the likelihood of removing the edge, plus some noise to break ties
                    reward = old_sol.accumulated_reward + prob * 1.0 + 1e-6 * rng.random()/edge_time_cost
                    current_val = old_sol.val
                #print(f"acc reward: {old_sol.accumulated_reward}, probability: {probability}, old_sol.value {old_sol.value}, current val {res["n_stations"]}")
                #For noisy heuristics, new value could be higher than old value, even if problem is a relaxation
                best_value = min(current_val, old_sol.value) 
                remaining_budget -= edge_time_cost
                if not old_sol.query_cost: #query cost is the cost of the first query
                    q_cost = edge_time_cost
                else:
                    q_cost = old_sol.query_cost
                sol = Solution(reward,prob,best_value, q_cost,remaining_budget,new_removed)
                if sol >= best_sol:
                    best_sol = sol
                elites.add(sol) #Adds solution if higher reward, otherwise ignores it
        if c_d > 1:
            if elites.same_first_edge():
                
                # print("elites agree on next move. proceeding to query")
                # print(elites.get_elites())
                break
        queue = elites.get_elites(sort_elites=True)
    obj = best_sol.accumulated_reward
    edges = best_sol.edges
    t_cost = best_sol.query_cost
    return edges[0], obj, t_cost

#best_first_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model,**new_kwargs):
def beam_search_ml( orig_salbp, G_max_close_orig,G_min, ml_model, ml_config={},remaining_budget=1e6,beam_config = {"width":1, "depth":1} , discount_factor = 1,**mhkwargs):
    width = beam_config["width"]
    depth = beam_config["depth"]
    G_max_close= G_max_close_orig.copy()
    G_max_red = nx.transitive_reduction(G_max_close)
    G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
    history = set() #Hash table of previous results
    best_sol = Solution(0,1,None,None, remaining_budget,[])
    queue = [best_sol]  # Queue of Solution(reward,probablity, val, removed_edges, first query cost) 

    for c_d in range(depth):
        elites = EliteSet(width)
        while len(queue)  > 0:
            print("Here is the depth and width", depth, " ", width)
            print("Here is the queue", queue)
            old_sol= queue.pop(0)
            remaining_budget = old_sol.remaining_budget
            to_remove = old_sol.edges.copy()
            G_max_red = remove_edges(G_max_close, to_remove)   
            edges = get_possible_edges(G_max_red, G_min, remaining_budget)
            edge_res = best_first_ml_choice_edge(edges,orig_salbp, G_max_red, ml_model, ml_config, top_n=width)
            if  edge_res: #No available edges to query can be due to time limit
                state_prob = old_sol.state_probability 
                for edge,_, contribution,_, t_cost in edge_res:
                    new_removed = to_remove+ [(edge[0], edge[1])]
                    #This is to avoid repeat computations (i.e. e1->e2, e2->e1)
                    edge_set = frozenset(new_removed)
                    if edge_set in history:
                        continue
                    # Store the current removal sequence
                    history.add(edge_set)
                    #We do not have real objective value, here we are maximizing the sum of the probabilities
                    #Contribution is (prob_edge_exist*prob_edge_contributes)/query_time
                    reward = old_sol.accumulated_reward + discount_factor*contribution* state_prob
                    new_budget = remaining_budget - t_cost
                    if not old_sol.query_cost: #query cost is the cost of the first query
                        q_cost = t_cost
                    else:
                        q_cost = old_sol.query_cost
                    sol = Solution(reward,state_prob,0,q_cost,new_budget, new_removed)
                    if sol > best_sol:
                        best_sol = sol
                    elites.add(sol)
                #sorting elites so we consider the sequence of edges with the higher accumulated reward first. 
                #Otherwise the history could accidentally prevent us from doing the better sequence
        queue = elites.get_elites(sort_elites=True)
    
 
        
        if c_d >= 1:
            if elites.same_first_edge():
                # print("elites agree on next move. proceeding to query")
                # print(elites.get_elites(sort_elites=True))
                break

    obj = best_sol.accumulated_reward
    edges = best_sol.edges
    t_cost = best_sol.query_cost
    return edges[0], obj, t_cost
