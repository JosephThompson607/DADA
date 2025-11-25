import networkx as nx
import numpy as np
from constraint_disc.beam_search import remove_edges, remove_edges_2
from ml_search import *
from set_new_edges import *
import random
import time
    
#     return phi, edge_data
def calc_phi_mh(orig_salbp, G_max_close_orig, G_max_red_orig, edges, mh,remaining_budget, old_value = 0, mode='lstd_prob', **mhkwargs):
    att = 0
    edge_data = {}
    if mode =="lstd_mh": #Safety, hopefully not necessary
        G_max_close = G_max_close_orig.copy()
        G_max_red = G_max_red_orig.copy()
    for i, edge in enumerate(edges):

        # Store the current removal sequence
        # Create list without current item
        edge_prob = edge[2]
        edge_cost = edge[3]

        if mode =="lstd_prob":
            reward = 1 * edge_prob/edge_cost
            n_stations = 0
            value_delta = edge_prob
        elif mode == "lstd_weight":
            reward = (orig_salbp['task_times'][edge[0]] + orig_salbp['task_times'][edge[1]]) * edge_prob/edge_cost
            n_stations = 0
            value_delta = orig_salbp['task_times'][edge[0]] + orig_salbp['task_times'][edge[1]]
            
        elif mode == "lstd_mh":
            #This is to avoid repeat computations (i.e. e1->e2, e2->e1)
            new_removed = [edge]
            G_max_close, G_max_red, added_edges = remove_and_expand(G_max_close, G_max_red, new_removed)
            new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
            res = mh(new_salbp, **mhkwargs)
            n_stations = res['n_stations']
            value_delta = max(0, (old_value - n_stations))
            reward =  edge_prob * value_delta/edge_cost
            #Resets G_max_red and G_max_close
            new_removed = [(edge[0], edge[1], {'prob':edge[2], 't_cost':edge[3]})]
            reinsert_edge(new_removed, added_edges, G_max_red, G_max_close)

        att += reward

        #saves dict for info
        edge_data[edge] = {'edge':edge, 'reward':reward, 'value':n_stations, 'value_delta': value_delta, 'edge_prob':edge_prob, 'edge_cost':edge_cost}
    future_budget = remaining_budget
    phi = np.zeros(5)
    phi[0] = att
    phi[1] = att * np.sqrt(future_budget)
    phi[2] = att * future_budget
    phi[3] = np.sqrt(future_budget)
    phi[4] = future_budget

    return phi,edge_data


def calc_phi_no_luck(phi_0, edge_prob, edge_weight, edge_cost, remaining_time):
    """Calculates phi for when the query fails"""
    phi_no_luck = phi_0.copy()

    att_no_luck = phi_0[0] - edge_prob*(edge_weight/edge_cost)
    phi_no_luck[0] = att_no_luck
    phi_no_luck[1] = np.sqrt(remaining_time) * att_no_luck
    phi_no_luck[2] = att_no_luck * remaining_time
    phi_no_luck[3] = np.sqrt(remaining_time)
    phi_no_luck[4] = remaining_time
    return phi_no_luck

def calc_phi_ml(orig_salbp, G_max_close, G_max_red, edges, ml_model, ml_config, remaining_budget, **_):
    profile = {}
    t_start = time.perf_counter()
    
    # Predictor
    t0 = time.perf_counter()
    edge_df = predictor(orig_salbp, G_max_close, G_max_red, ml_model, ml_config, **_)
    profile['predictor'] = time.perf_counter() - t0
    
    # Filter for valid edges
    t0 = time.perf_counter()
    edge_prob_df = filter_for_valid_edges(edges, edge_df.copy())
    profile['filter_valid_edges'] = time.perf_counter() - t0
    
    # Reward calculation
    t0 = time.perf_counter()
    edge_prob_df['reward'] = edge_prob_df['precedent_prob'] * edge_prob_df['pred_val_prob'] / edge_prob_df['t_cost']
    att = edge_prob_df['reward'].sum()
    profile['reward_calc'] = time.perf_counter() - t0
    
    # Phi computation
    t0 = time.perf_counter()
    future_budget = max(0, remaining_budget)
    phi = np.zeros(5)
    phi[0] = att
    phi[1] = att * np.sqrt(future_budget)
    phi[2] = att * future_budget
    phi[3] = np.sqrt(future_budget)
    phi[4] = future_budget
    profile['phi_computation'] = time.perf_counter() - t0
    
    profile['total'] = time.perf_counter() - t_start
    
    #Print profiling results
    # print(f"Profiling calc_phi_ml:")
    # for key, val in profile.items():
    #     print(f"  {key}: {val*1000:.3f} ms")
    
    return phi, edge_prob_df


# def calc_phi_ml(orig_salbp,G_max_close, G_max_red, edges, ml_model,ml_config,remaining_budget,  **_):
#     edge_df = predictor(orig_salbp, G_max_close, G_max_red, ml_model, ml_config,**_)
#     edge_prob_df = filter_for_valid_edges(edges, edge_df.copy())
#     edge_prob_df['reward'] =  edge_prob_df['precedent_prob']* edge_prob_df['pred_val_prob']/edge_prob_df['t_cost']
#     att = edge_prob_df['reward'].sum()
#     #     att += reward
#     future_budget = max(0,remaining_budget)
#     phi = np.zeros(5)
#     phi[0] = att
#     phi[1] = att * np.sqrt(future_budget)
#     phi[2] = att * future_budget
#     phi[3] = np.sqrt(future_budget)
#     phi[4] = future_budget
#     return phi, edge_prob_df
   


def train_lstd(orig_salbp, G_max_close_orig, G_min_orig, max_budget, mh, mode='lstd_prob', seed=42, discount_factor=0.95,prev_val=0,lambda_reg=1e-6,ml_model=None, ml_config=None,n_episodes=50, **mhkwargs):
    theta = np.zeros(5)
    A = np.zeros((5, 5))
    b = np.zeros(5)
    print('training. lstd. mode: ', mode)
    rng = random.Random(seed)
    G_max_red_orig = nx.transitive_reduction(G_max_close_orig)
    G_max_red_orig.add_edges_from((u, v, G_max_close_orig.edges[u, v]) for u, v in G_max_red_orig.edges)
    edges_orig = get_possible_edges(G_max_red_orig, G_min_orig, max_budget)
    for trial in range(n_episodes):
        start = time.time()
        G_min = G_min_orig.copy()
        remaining_budget = max_budget
        G_max_close = G_max_close_orig.copy()
        G_max_red = G_max_red_orig.copy()
        edges = edges_orig.copy()
        if mode == 'lstd_ml':
            phi_0, edge_data = calc_phi_ml(orig_salbp, G_max_close, G_max_red, edges, ml_model,ml_config,remaining_budget)
        else:
            # print("here are the mh kwargs train lstd", mhkwargs)
            phi_0, edge_data = calc_phi_mh(orig_salbp, G_max_close,G_max_red, edges, mh, remaining_budget, old_value=prev_val, mode=mode, **mhkwargs)
        # print(f"here is phi_0, {phi_0}, running episode")
        A, b = run_episode(
            orig_salbp, G_max_close, G_max_red, G_min, edges, 
            remaining_budget, phi_0, theta, A, b, mh, mode, 
            discount_factor, rng,ml_model=ml_model, ml_config=ml_config,prev_val=prev_val,edge_data=edge_data, **mhkwargs
        )
        
        # update theta
        #1e-6 #regularization terms to avoid singular matrices
        theta = np.linalg.inv(A+ lambda_reg * np.eye(5)) @ b
        end = time.time()
        print(" trial ", trial, " This is theta", theta, " trial time is ", end-start)
    return theta


def run_episode(orig_salbp, G_max_close, G_max_red, G_min, edges, 
                remaining_budget, phi_0, theta, A, b, mh, mode, 
                discount_factor, rng,ml_model,ml_config, prev_val = None,edge_data=None, **mhkwargs):
    """
    Run a single episode of the LSTD algorithm, selecting edges and updating A and b matrices.
    
    Returns:
        tuple: Updated (A, b) matrices
    """

    while edges and remaining_budget > 0:
        start_time = time.time()
        # Find best edge
        if mode in ( "lstd_mh", "lstd_prob", "lstd_weight") :
            best_edge, best_prob, best_weight,best_val, best_time = select_best_edge(
                edges, orig_salbp, G_max_close,G_max_red, G_min, mh, 
                remaining_budget, phi_0, theta, mode,prev_val, edge_data,**mhkwargs
            )
            
            
        elif mode == "lstd_ml":
            best_edge, best_prob, best_weight,_, best_time = select_best_edge_ml(
                edges, orig_salbp, G_max_close, G_max_red, G_min, 
                remaining_budget, phi_0, theta, ml_model=ml_model, ml_config=ml_config,edge_data=edge_data, **mhkwargs
            )
            best_edge = (best_edge[0],best_edge[1], best_prob, best_time)
            best_val=prev_val #Don't keep track the objective value
        else:
            print("Error: No vaild mode selected")
        edge_selection_time = time.time()
        #print("time to select best edge: ", edge_selection_time - start_time )
        # print("Best edge", best_edge, "best_weight", best_weight, "best prob", best_prob)
        # update b
        b = b + phi_0 * best_prob * best_weight
        
        # State update - random state transition
        G_max_close, G_max_red, G_min, success = uncertain_query_prec_set(
            G_max_close, G_max_red, G_min, best_edge, rng
        )
        update_time = time.time()
        #print(" edge update time ", update_time -edge_selection_time)
        if success: #If edges successfully removed, we have the new objective value
            prev_val = best_val
        # update budget
        remaining_budget -= best_time
        #update edge set to choose from
        edges = get_possible_edges(G_max_red, G_min, remaining_budget=remaining_budget)
        edge_find_time = time.time()
        #print("edge find time ", edge_find_time - update_time)
        # update A and phi_0
        if mode == 'lstd_ml':
            phi_1,edge_data = calc_phi_ml(orig_salbp, G_max_close, G_max_red, edges, ml_model, ml_config, remaining_budget)
        else:
            phi_1, edge_data = calc_phi_mh(orig_salbp, G_max_close, G_max_red, edges, mh, remaining_budget, old_value=prev_val, mode=mode, **mhkwargs)
        step = phi_0 - discount_factor * phi_1
        # print(f"phi_0 {phi_0}, phi_1 {phi_1} ")
        #print("Here is the step", step)
        A = A + np.outer(phi_0, step)
        phi_0 = phi_1
        phi_and_math_time = time.time()
        #print("time for updating linealg", phi_and_math_time - edge_find_time)
    return A, b


def select_best_edge(edges, orig_salbp, G_max_close_orig,G_max_red_orig, G_min, mh, 
                     start_time, phi_0, theta, mode, prev_val, edge_data=None,**mhkwargs):
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

    time_copy = 0
    time_transitive = 0
    time_mh = 0
    time_phi = 0
    G_max_close = G_max_close_orig.copy() #Copying for safety, hopefully unecessary
    G_max_red = G_max_red_orig.copy()
    for i, edge in enumerate(edges):
        t1 = time.perf_counter()

        new_removed = [edge]
        G_max_close, G_max_red, added_edges = remove_and_expand(G_max_close, G_max_red, new_removed)
        time_transitive += time.perf_counter() - t1
        t3 = time.perf_counter()
        remaining_time = start_time-edge[3]
        new_edges = get_possible_edges(G_max_red, G_min,remaining_time )
        if edge_data and edge in edge_data.keys():
           weight = edge_data[edge]['value_delta']
           val = edge_data[edge]['value']
        else:
            if mode == 'lstd_mh':
                new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
                res = mh(new_salbp, **mhkwargs)
                weight = max(0, prev_val - res['n_stations']) 
                val = min(prev_val,res['n_stations']) #Sometimes mh can give a greater value with less constraints
            elif mode == 'lstd_weight':

                weight = orig_salbp['task_times'][edge[0]] + orig_salbp['task_times'][edge[1]]

            else:
                weight = 1
                val = 0
        time_mh = time.perf_counter() - t3
        t4 = time.perf_counter()
        phi, _= calc_phi_mh(orig_salbp, G_max_close,G_max_red, new_edges, mh, remaining_time, old_value=val, mode=mode, **mhkwargs)
        phi_no_luck = calc_phi_no_luck(phi_0, edge[2], weight, edge[3], remaining_time)
        reward =max(0, edge[2] * (weight + np.dot(phi, theta)) + (1 - edge[2]) * np.dot(phi_no_luck, theta))
        #Reset graph
        new_removed = [(edge[0], edge[1], {'prob':edge[2], 't_cost':edge[3]})]
        reinsert_edge(new_removed, added_edges, G_max_red, G_max_close)
        time_phi += time.perf_counter()-t4
        #Check if best reward
        if reward >= best_reward:
            best_edge = edge
            best_prob = edge[2]
            best_weight = weight
            best_reward = reward
            best_val = val
            best_time = edge[3]
    # print("\n🔍 TIMING BREAKDOWN:")
    # print(f"   Transitive closure:   {time_transitive:.4f}s  ")
    # print(f"   Metaheuristic (mh):   {time_mh:.4f}s")
    # print(f"   Phi calculations:     {time_phi:.4f}s ⚠️ LIKELY BOTTLENECK")
    # total = time_copy + time_transitive + time_mh + time_phi
    # print(f"   Total tracked:        {total:.4f}s")
    
    return best_edge, best_prob, best_weight,best_val, best_time



def select_best_edge_ml(edges, orig_salbp, G_max_close, G_max_red, G_min,
                     start_time, phi_0, theta, ml_model, ml_config, edge_data=None,**mhkwargs):
    """
    Select the best edge based on expected reward. Used in LSTD
    
    Returns:
        tuple: (best_edge, best_prob, best_weight, best_val, best_time)
    """
    best_edge = []
    best_reward = 0
    best_weight = 0
    best_prob = 0
    best_time = 1
    #First, get the weights based on the current state
    if edge_data is not None:
        edge_res = list(zip(edge_data['edge'], edge_data['pred_val_prob'],edge_data['reward'], edge_data['precedent_prob'], edge_data['t_cost']))
    else:
        edge_res = best_first_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model,ml_config, top_n=len(edges), **_)
    #Now test the weights of the next state
    for edge,weight,_,proba, t_cost in edge_res:
        # Create view without current edge
        new_removed = [(edge[0], edge[1])]
        G_max_close, G_max_red, added_edges = remove_and_expand(G_max_close, G_max_red, new_removed)

        remaining_time = start_time-t_cost
        new_edges = get_possible_edges(G_max_red, G_min,remaining_time )
        phi,_ = calc_phi_ml(orig_salbp, G_max_close, G_max_red, new_edges, ml_model,ml_config,remaining_time)
        phi_no_luck = calc_phi_no_luck(phi_0,  proba , weight, t_cost, remaining_time)
        reward =max(0,  proba *(weight + np.dot(phi, theta)) + (1 - proba) * np.dot(phi_no_luck, theta))
        #Reset graph
        new_removed = [(edge[0], edge[1], {'prob':proba, 't_cost':t_cost})]
        reinsert_edge(new_removed, added_edges, G_max_red, G_max_close)

        if reward >= best_reward:
            best_reward = reward
            best_edge = edge
            best_prob = proba
            best_weight = weight 
            best_val = 0 #We don't really keep track of values here
            best_time =  t_cost

    return best_edge, best_prob, best_weight,best_val, best_time


def lstd_search(orig_salbp, G_max_close_orig, G_min, mh,remaining_budget, theta,prev_val, mode="lstd_prob", ml_config=None, ml_model=None, **mhkwargs ):
    G_max_close = G_max_close_orig.copy()
    G_max_red = nx.transitive_reduction(G_max_close)
    G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
    edges = get_possible_edges(G_max_red, G_min, remaining_budget=remaining_budget)
    if mode == "lstd_ml":
        #Here phi_0 is just used to generate the case when the edge selection fails, but can reuse the results
        phi_0, edge_results = calc_phi_ml(orig_salbp,G_max_close, G_max_red, edges, ml_model, ml_config, remaining_budget)
        best_edge, best_prob,_, _, best_time =select_best_edge_ml(edges, orig_salbp, G_max_close, G_max_red, G_min, remaining_budget, phi_0,theta,ml_model=ml_model, ml_config=ml_config, edge_data = edge_results)
    else:
        phi_0,edge_results = calc_phi_mh(orig_salbp, G_max_close, G_max_red, edges, mh,remaining_budget, old_value = 0, mode=mode, **mhkwargs)
        best_edge, best_prob,_, _, best_time =select_best_edge(edges, orig_salbp, G_max_close,G_max_red, G_min, mh, remaining_budget, phi_0,theta, mode=mode,prev_val=prev_val, edge_data=edge_results)
    return best_edge, best_prob, best_time