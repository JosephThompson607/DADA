import networkx as nx
import numpy as np
from constraint_disc.beam_search import remove_edges
from set_new_edges import *
import random

def calc_phi_mh(orig_salbp, G_max_close, G_min,mh,remaining_budget, old_value = 0, mode='lstd_prob', **mhkwargs):
    G_max_red = nx.transitive_closure(G_max_close)
    edges = get_possible_edges(G_max_red, G_min)
    att = 0
    for i, edge in enumerate(edges):
                #This is to avoid repeat computations (i.e. e1->e2, e2->e1)
        new_removed = [(edge[0], edge[1])]
        # Store the current removal sequence
        # Create list without current item
        if len(edge) ==3:
            edge_prob = edge[2]
            edge_cost = 1
        elif len(edge) ==4:
            edge_prob = edge[2]
            edge_cost = edge[3]
        else:
            edge_prob = 1
            edge_cost = 1
        if mode =="lstd_prob":
            reward = 1 * edge_prob/edge_cost
            
        elif mode == "lstd_mh":
            G_max_red = remove_edges(G_max_close, new_removed)

            new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
            res = mh(new_salbp, **mhkwargs)
            reward =  max( 0, edge_prob*(old_value - res['n_stations']))/edge_cost
        att += reward
    future_budget = max(0,remaining_budget- edge_cost)
    phi = np.zeros(5)
    phi[0] = att
    phi[1] = att * np.sqrt(future_budget)
    phi[2] = att * future_budget
    phi[3] = np.sqrt(future_budget)
    phi[4] = future_budget
    return phi
   


def train_lstd(orig_salbp, G_max_close_orig, G_min_orig, max_budget, mh, mode='lstd_prob', m=50, seed=42, discount_factor=0.95,prev_val=0, **mhkwargs):
    theta = np.zeros(5)
    A = np.zeros((5, 5))
    b = np.zeros(5)
    rng = random.Random(seed)
    for trial in range(m):
        G_min = G_min_orig.copy()
        remaining_budget = max_budget
        G_max_close = G_max_close_orig.copy()
        G_max_red = nx.transitive_reduction(G_max_close)
        G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
        edges = get_possible_edges(G_max_red, G_min)
        phi_0 = calc_phi_mh(orig_salbp, G_max_close, G_min, mh, remaining_budget, old_value=prev_val, mode=mode, **mhkwargs)
        
        A, b = run_episode(
            orig_salbp, G_max_close, G_max_red, G_min, edges, 
            remaining_budget, phi_0, theta, A, b, mh, mode, 
            discount_factor, rng,prev_val, **mhkwargs
        )
        
        # update theta
        lambda_reg = 1e-6 #regularization terms to avoid singular matrices
        theta = np.linalg.inv(A+ lambda_reg * np.eye(5)) @ b
        print("This is theta", theta)
    
    return theta


def run_episode(orig_salbp, G_max_close, G_max_red, G_min, edges, 
                remaining_budget, phi_0, theta, A, b, mh, mode, 
                discount_factor, rng, prev_val = None,**mhkwargs):
    """
    Run a single episode of the LSTD algorithm, selecting edges and updating A and b matrices.
    
    Returns:
        tuple: Updated (A, b) matrices
    """

    while edges and remaining_budget > 0:
        print("Here is the prev val", prev_val)
        # Find best edge
        best_edge, best_prob, best_weight,best_val, best_time = select_best_edge(
            edges, orig_salbp, G_max_close, G_min, mh, 
            remaining_budget, phi_0, theta, mode,prev_val, **mhkwargs
        )
        
        print("Best edge", best_edge, "best_weight", best_weight, "best prob", best_prob)
        # update b
        b = b + phi_0 * best_prob * best_weight
        
        # State update - random state transition
        G_max_close, G_max_red, G_min, success = uncertain_query_prec_set(
            G_max_close, G_max_red, G_min, best_edge, rng
        )
        if success:
            prev_val = best_val
        # update budget
        remaining_budget -= best_time
        
        # update available edges
        edges = get_possible_edges(G_max_red, G_min)
        
        # update A and phi_0
        phi_1 = calc_phi_mh(orig_salbp, G_max_close, G_min, mh, remaining_budget, old_value=prev_val, mode=mode, **mhkwargs)
        step = phi_0 - discount_factor * phi_1
        print(f"phi_0 {phi_0}, phi_1 {phi_1} ")
        print("Here is the step", step)
        A = A + np.outer(phi_0, step)
        phi_0 = phi_1
    
    return A, b


def select_best_edge(edges, orig_salbp, G_max_close, G_min, mh, 
                     remaining_budget, phi_0, theta, mode, prev_val, **mhkwargs):
    """
    Select the best edge based on expected reward. Used in LSTD
    
    Returns:
        tuple: (best_edge, best_prob, best_weight, best_time)
    """
    best_edge = []
    best_reward = 0
    best_weight = 0
    best_prob = 0
    best_time = 1
    for edge in edges:
        new_removed = [(edge[0], edge[1])]
        G_max_close_test = G_max_close.copy()
        G_max_close_test.remove_edges_from(new_removed)
        if mode == 'lstd_mh':
            G_max_red = nx.transitive_closure(G_max_close_test)
            new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
            res = mh(new_salbp, **mhkwargs)
            
            weight = max(0, prev_val - res['n_stations']) + 1e-6
            val = min(prev_val,res['n_stations']) #Sometimes mh can give a greater value with less constraints
        else:
            weight = 1
            val = 0
        
        phi = calc_phi_mh(orig_salbp, G_max_close_test, G_min, mh, remaining_budget, old_value=val, mode=mode, **mhkwargs)
        reward =max(0, edge[2] * (weight + np.dot(phi, theta)) + (1 - edge[2]) * np.dot(phi_0, theta))
        if reward >= best_reward:
            best_edge = edge
            best_prob = edge[2]
            best_weight = weight
            best_reward = reward
            best_val = val
            if len(edge) == 4:
                best_time = edge[3]
    
    return best_edge, best_prob, best_weight,best_val, best_time

def lstd_search_mh(orig_salbp, G_max_close_orig, G_min, mh,remaining_budget, theta,prev_val, mode="lstd_prob", **mhkwargs ):
    G_max_close = G_max_close_orig.copy()
    G_max_red = nx.transitive_reduction(G_max_close)
    G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
    edges = get_possible_edges(G_max_red, G_min)
    phi_0 = calc_phi_mh(orig_salbp, G_max_close, G_min,mh,remaining_budget, old_value = 0, mode=mode, **mhkwargs)
    best_edge, best_prob,_, _, _ =select_best_edge(edges, orig_salbp, G_max_close, G_min, mh, remaining_budget, phi_0,theta, mode=mode,prev_val=prev_val)
    return best_edge, best_prob