import re
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import subprocess
import sys
import time
import os
import xgboost as xgb
import copy

sys.path.append('src')
from metrics.graph_features import calculate_order_strength

from metrics.node_and_edge_features import randomized_kahns_algorithm
from data_prep import albp_to_features
from alb_instance_compressor import *
from SALBP_solve import salbp1_bbr_call, salbp1_prioirity_solve, salbp1_vdls_dict, salbp1_mhh_solve
build_dir = '/Users/letshopethisworks2/CLionProjects/SALBP_ILS/cmake-build-python_interface/'
sys.path.insert(0, build_dir)
build_dir_2 = '/home/jot240/DADA/SALBP_ILS/build/'
sys.path.insert(0, build_dir_2)
import ILS_ALBP
import multiprocessing


def nx_to_albp(G,C=1000):
    """
    Convert NetworkX DiGraph back to SALBP dictionary format
    
    Args:
        G: NetworkX DiGraph object
        name: String name for the SALBP problem
    
    Returns:
        dict: SALBP dictionary with keys: 'name', 'task_times', 'precedence_relations'
    """

    task_times = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        if 'task_time' in node_data:
            task_time = node_data['task_time']
        else:
            print(f"Warning: Node {node} has no task_time attribute, using default value 1")
            task_time = 1
        task_times[str(node)] = task_time
    # Extract precedence relations from edges
    precedence_relations = list(G.edges())
    
    # Create SALBP dictionary
    SALBP_dict = {
        'n_tasks': len(task_times.keys()),
        'task_times': task_times,
        'precedence_relations': precedence_relations,
        'cycle_time':C
    }
    
    return SALBP_dict


def albp_to_nx(SALBP_dict):
    G = nx.DiGraph()
    G.add_nodes_from(SALBP_dict["task_times"].keys())
    G.add_edges_from(SALBP_dict["precedence_relations"])
    nx.set_node_attributes(G, {i: {'task_time': SALBP_dict['task_times'][str(i)]} for i in G.nodes})
    return G
def plot_salbp_dict(SALBP_dict):
    G = albp_to_nx(SALBP_dict)
    #prints the edges

    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    #prints the edges from the graph
    nx.draw(G,pos, with_labels = True)
    plt.show()

def plot_salbp_graph(G):

    P = nx.nx_pydot.to_pydot(G)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    #prints the edges from the graph
    nx.draw(G,pos, with_labels = True)
    plt.show()

def albp_to_sequences(albp, n_sequences, seed=42):
    G = albp_to_nx(albp)
    return randomized_kahns_algorithm(G, n_runs=n_sequences, seed=seed)

def seqs_to_dag(orig_sequences):
    feasible_sequences = orig_sequences.copy()
    G = nx.DiGraph()
    #Add nodes
    seq1 = feasible_sequences.pop(0)
    n = len(seq1)
    G.add_nodes_from(seq1)
    #Initial graph
    for i in range(1, n+1):
        i_ind = seq1.index(str(i))
        for j in range(1,n+1):
            j_ind = seq1.index(str(j))
            if (i !=j) and (i_ind < j_ind):
                G.add_edge(str(i), str(j))
    for p in feasible_sequences:
        for i in range(1, n+1):
            i_ind = p.index(str(i))
            for j in range(1, n+1):
                j_ind = p.index(str(j))
                if G.has_edge(str(i), str(j)) and j_ind < i_ind:
                    G.remove_edge(str(i), str(j))
    


    return G

def naive_query_prec_set(G_max_close, G_max_red, G_min, G_true, edge):
    if G_max_close.has_edge(edge[0], edge[1]) and not G_true.has_edge(edge[0], edge[1]):
        G_max_close.remove_edge(edge[0], edge[1])
        G_max_red= nx.transitive_reduction(G_max_close)
    elif G_max_close.has_edge(edge[0], edge[1]) and G_true.has_edge(edge[0], edge[1]):
        G_min.add_edge(edge[0], edge[1])
    else:
        print(f"edge {edge} not in transitive closure")
        
    return G_max_close, G_max_red, G_min

def focused_query_prec_set(G_max_close, G_max_red, G_min, G_true, edge):
    """This function only removes an edge if it is part of the transitive reduction of 
        Ē but not in the real precedence constraints"""
    successful_removal = False
    if G_max_red.has_edge(edge[0], edge[1]) and not G_true.has_edge(edge[0], edge[1]):
        G_max_close.remove_edge(edge[0], edge[1])
        successful_removal = True
        G_max_red= nx.transitive_reduction(G_max_close)
    elif G_max_red.has_edge(edge[0], edge[1]) and G_true.has_edge(edge[0], edge[1]):
        G_min.add_edge(edge[0], edge[1])
    else:
        print(f"edge {edge} not in transitive reduction")
        print(list(G_max_red.edges()))
    return G_max_close, G_max_red, G_min, successful_removal




def get_possible_edges(G_max_red, G_min):
    G_min_close = nx.transitive_closure(G_min)
    candidates = []
    for edge in  G_max_red.edges():
        if not G_min_close.has_edge(edge[0], edge[1]):
            candidates.append(edge)
    return candidates


def non_repeating_strat(G_max_close, G_min, G_true, n_queries ):
    node_attrs = dict(G_true.nodes(data=True))
    nx.set_node_attributes(G_max_close, node_attrs) 
    G_max_red = nx.transitive_reduction(G_max_close)
    nx.set_node_attributes(G_max_red, node_attrs) 
    for q in range(n_queries):
        edges = get_possible_edges(G_max_red, G_min)
        edge =  random.choice(edges)
        G_max_close, G_max_red, G_min,_ = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
    return G_max_red

def rename_nodes_topological(G):
    """
    Rename nodes to respect topological ordering
    
    Args:
        G: NetworkX DiGraph with string node names
    
    Returns:
        tuple: (new_graph, old_to_new_mapping, new_to_old_mapping)
    """
    # Get topological ordering
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXError:
        raise ValueError("Graph contains cycles - not a DAG")
    
    # Create mappings
    old_to_new = {}
    new_to_old = {}
    
    for new_index, old_node in enumerate(topo_order, 1):
        new_name = str(new_index)
        old_to_new[old_node] = new_name
        new_to_old[new_name] = old_node
    
    # Create new graph with renamed nodes
    new_graph = nx.relabel_nodes(G, old_to_new)
    
    return new_graph, old_to_new, new_to_old

def set_new_edges(G_max_red, orig_salbp):
    topo_G, old_to_new, new_to_old = rename_nodes_topological(G_max_red)
    new_salbp = copy.deepcopy(orig_salbp)
    new_task_times = {}

    for task, time in orig_salbp['task_times'].items():
        new_task_times[old_to_new[task]] = time
    new_salbp['precedence_relations'] = [[e1, e2] for (e1,e2) in topo_G.edges()]
    #Need to respect order in task times or else bbr solver will break =/
    new_salbp['task_times'] = {k: new_task_times[k] for k in sorted(new_task_times, key=lambda x: int(x))}
    return new_salbp, new_to_old



def non_repeating_strat_w_eval(G_max_close, G_min, orig_salbp, n_queries , ex_fp):
    G_true = albp_to_nx(orig_salbp)
    G_max_red = nx.transitive_reduction(G_max_close)
    order_strength = calculate_order_strength(G_max_red)
    test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=3)
    orig_n_stations = res['n_stations']
    n_stations = orig_n_stations
    bin_limit = res['bin_lb']
    query_vals = [{"n_stations":orig_n_stations, "OS":order_strength}]
    for q in range(n_queries):
        if n_stations == bin_limit:
            print(f"reached bpp lower bound terminating after {q} queries")
            break
        edges = get_possible_edges(G_max_red, G_min)

        edge =  random.choice(edges)
        G_max_close, G_max_red, G_min, _ = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
        order_strength = calculate_order_strength(G_max_red)
        test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=3, w_bin_pack=False)
        n_stations = res['n_stations']
        query_vals.append({"n_stations":n_stations, "OS":order_strength})

    return G_max_red, query_vals,bin_limit



def myopic_choice(edges, orig_salbp, G_max_red_orig, ex_fp , q_check_tl=3):
    G_max_red = G_max_red_orig.copy()
    best_edge = edges[0]
    remaining_edges =edges[1:]
    G_max_red.clear_edges()
    G_max_red.add_edges_from(remaining_edges)
    new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=q_check_tl)
   
    station_best = res['n_stations']
    for i, edge in enumerate(edges[1:]):
        if res['n_stations'] < station_best:
            station_best = res['n_stations']
            best_edge = edge
            if res['n_stations'] == res['bin_lb']:
                break
        # Create list without current item
        remaining_edges = edges[:i] + edges[i+1:]
        G_max_red.clear_edges()
        G_max_red.add_edges_from(remaining_edges)
        new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=q_check_tl)

    return best_edge, station_best


  
def greedy_choice(edges, orig_salbp, G_max_red_orig, ex_fp, q_check_tl=3 ):
    G_max_red = G_max_red_orig.copy()
    best_edge = edges[0]
    remaining_edges =edges[1:]
    G_max_red.clear_edges()
    G_max_red.add_edges_from(remaining_edges)
    new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=q_check_tl, w_bin_pack=False)
    station_best = res['n_stations']
    for i, edge in enumerate(edges[1:]):
        if res['n_stations'] < station_best:
            station_best = res['n_stations']
            best_edge = edge
            return best_edge, station_best     
        # Create list without current item
        remaining_edges = edges[:i] + edges[i+1:]
        G_max_red.clear_edges()
        G_max_red.add_edges_from(remaining_edges)
        new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=1, w_bin_pack=False)

    return best_edge, station_best        


def greedy_choice_mh(edges, orig_salbp, G_max_red_orig, mh, greedy=True,**mhkwargs ):
    G_max_red = G_max_red_orig.copy()
    best_edge = edges[0]
    remaining_edges =edges[1:]
    G_max_red.clear_edges()
    G_max_red.add_edges_from(remaining_edges)
    new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    #res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=1, w_bin_pack=False)
    res = mh(new_salbp, **mhkwargs)

    station_best = res['n_stations']


    for i, edge in enumerate(edges[1:]):
        if res['n_stations'] < station_best:
            station_best = res['n_stations']
            best_edge = edge
            if greedy:
                return best_edge, station_best     
        # Create list without current item
        remaining_edges = edges[:i] + edges[i+1:]
        G_max_red.clear_edges()
        G_max_red.add_edges_from(remaining_edges)
        new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        #res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=1, w_bin_pack=False)
        res = mh(new_salbp, **mhkwargs)
    return best_edge, station_best      


def predictor(orig_salbp, G_max_red, ml_model):
    #Take task times form base salbp problem and use them with the new edges
    test_salbp = copy.deepcopy(orig_salbp)
    test_salbp['precedence_relations'] = list(G_max_red.edges())
    edge_features = albp_to_features(test_salbp, salbp_type="salbp_1", cap_constraint = None, n_random=100)
    edge_names = edge_features['edge']
    continuous_features_edge = continuous_features_edge = [
'parent_weight', 'parent_pos_weight',
       'child_weight', 'child_pos_weight', 'neighborhood_min',
       'neighborhood_max', 'neighborhood_avg', 'neighborhood_std',
       'parent_in_degree', 'parent_out_degree', 'child_in_degree',
       'child_out_degree', 'chain_avg', 'chain_min', 'chain_max', 'chain_std',
       'edge_data_time', 'rw_mean_total_time', 'rw_mean_min_time',
       'rw_mean_max_time', 'rw_mean_n_unique_nodes',
       'rw_min', 'rw_max', 'rw_mean', 'rw_std', 'rw_n_unique_nodes', 'child_rw_mean_total_time', 'child_rw_mean_min_time',
       'child_rw_mean_max_time', 'child_rw_mean_n_unique_nodes', 'child_rw_min', 'child_rw_max',
       'child_rw_mean', 'child_rw_std', 'child_rw_n_unique_nodes', 'priority_min_gap', 'priority_max_gap',
       'random_spread', 'random_coefficient_of_variation', 'random_avg_gap',
       'random_min_gap', 'random_max_gap', 
        'ti_size',
        'prec_bias',
        'stage_difference',
        'prec_strength',
        'load_parent_mean' ,
        'load_parent_max' ,
        'load_parent_min' ,
        'load_parent_std' ,
        'load_child_mean' ,
        'load_child_max' ,
        'load_child_min' ,
        'load_child_std' ,
       'min_div_c', 'max_div_c',  'std_div_c','avg_div_c',
       'order_strength', 'average_number_of_immediate_predecessors',
       'max_degree', 'max_in_degree', 'max_out_degree', 'divergence_degree',
       'convergence_degree',  'share_of_bottlenecks',
        'avg_chain_length',
       'nodes_in_chains', 'stages_div_n', 'n_isolated_nodes',
       'share_of_isolated_nodes', 'n_tasks_without_predecessors',
       'share_of_tasks_without_predecessors', 'avg_tasks_per_stage',
] 
    X_all = edge_features[continuous_features_edge]
    y_prob = ml_model.predict_proba(X_all)[:, 1]
    edge_prob_df = pd.DataFrame({
        'edge': edge_names,
        'predicted_probability': y_prob
    })
    return edge_prob_df

def select_best_edge(edge_prob_df, valid_edges):
    """
    Filter edge_prob_df to only include edges in valid_edges,
    then return the edge with the highest predicted probability.
    """
    # Filter DataFrame by valid edges
    valid_edges = set([(str(e1),str(e2)) for (e1,e2) in valid_edges])
    edge_prob_df['valid'] = edge_prob_df['edge'].apply(lambda x: (str(x[0]), str(x[1])) in valid_edges)
    filtered = edge_prob_df[edge_prob_df['valid']==True]
    if filtered.empty:
        print("Error: No edges in the probability dataframe ")
        return None  # or raise an exception if that’s unexpected

    # Select edge with max probability
    best_row = filtered.loc[filtered['predicted_probability'].idxmax()]
    return best_row['edge'], best_row['predicted_probability']

def myopic_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model,**new_kwargs):
    '''Selects the edge with the highest probability of reducing the objective value'''

    prob_df = predictor(orig_salbp, G_max_red, ml_model)
    best_edge = select_best_edge(prob_df, edges)

    return best_edge




def greedy_reduction(G_max_close, G_min,  orig_salbp, n_queries , ex_fp, mh, q_check_tl=3, selector_method='myopic',ml_model = None, seed=42,**mhkwargs):
    G_true = albp_to_nx(orig_salbp)
    start = time.time()
    G_max_red = nx.transitive_reduction(G_max_close)
    order_strength = calculate_order_strength(G_max_red)
    test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    orig_bbr = False
    if "BBR-for-SALBP1/" in ex_fp:
        orig_bbr = True
    res = salbp1_bbr_call(test_salbp,ex_fp, 1, w_bin_pack=True, time_limit=q_check_tl, orig_bbr=orig_bbr)
    init_sol = res['task_assignments']
    new_kwargs = {**mhkwargs, 'initial_solution':init_sol, 'ex_fp':ex_fp, 'orig_bbr':orig_bbr}
    orig_n_stations = res['n_stations']
    n_stations = orig_n_stations
    bin_limit = res['bin_lb']
    random.seed(seed)

    #print(f"SALBP number of stations before : {orig_n_stations}, ellapsed time {time.time()- start}, bin lb {bin_limit} OS {order_strength}")
    query_vals = [{"n_stations":orig_n_stations, "OS":order_strength}]
    for q in range(n_queries): 
        if n_stations == bin_limit:
            print(f"reached bpp lower bound terminating after {q} queries")
            break
        
        edges = get_possible_edges(G_max_red, G_min)
        random.shuffle(edges)
        greedy_time = time.time()
        if selector_method == 'myopic':
            edge, n_stations= greedy_choice_mh(edges, orig_salbp, G_max_red, mh, greedy=False,**new_kwargs)
        elif selector_method == 'ml':
            edge, probability = myopic_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model )
        else:
            edge, n_stations= greedy_choice_mh(edges, orig_salbp, G_max_red, mh, **new_kwargs)
        G_max_close, G_max_red, G_min, _ = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
        order_strength = calculate_order_strength(G_max_red)
        test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=q_check_tl, orig_bbr=orig_bbr)
        n_stations = res['n_stations']
        query_vals.append({"n_stations":n_stations, "OS":order_strength})
    return G_max_red, query_vals, bin_limit


def get_feasible_seq(albp, n_seq, seed=42):
    topo_sorts= albp_to_sequences(albp, n_seq, seed=seed)
    feasible_sequences = []
    for i in topo_sorts:
        feasible_sequences.append(i['sorted_nodes'])
    return feasible_sequences


def do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, mh,selector_method, time_limit=1,seed=42, **mh_kwargs):
    G_max_close2 = G_max_close.copy()
          
    G_min = nx.DiGraph()
    G_min.add_nodes_from(albp_problem["task_times"].keys())
    r_time = time.time()
    G_max_red_random, random_query_vals, random_bin_limit =  greedy_reduction(G_max_close2, G_min, albp_problem, n_queries, ex_fp, mh, time_limit=time_limit,selector_method=selector_method,seed=seed, **mh_kwargs)
    r_time = time.time()-r_time
    os = [d["OS"] for d in random_query_vals]
    val = [d["n_stations"] for d in random_query_vals]
    return { 'time': r_time,  'OS':os,'query_values':val,  'bin_lb':random_bin_limit}



def constraint_elim(albp_problem, n_tries, ex_fp, save_folder, n_queries,n_start_sequences, selector_method, ml_model_fp, base_seed=None, fast_only=False):
    name = str(albp_problem['name']).split('/')[-1].split('.')[0]            
    with open(ml_model_fp, 'rb') as f:
        ml_model = pickle.load(f)
    res_list = []
    for attempt_ind in range(n_tries):
        #albp_problem = parse_alb("/Users/letshopethisworks2/Documents/phd_paper_material/DADA/test.alb")
        feasible_sequences = get_feasible_seq(albp_problem,n_start_sequences, seed=attempt_ind+base_seed)
        metadata = {'dataset':albp_problem['name'], 'name': name, 'n_queries':n_queries, 'attempt':attempt_ind, 'n_start_sequences':n_start_sequences, "dp_search_strategy":selector_method}
        G_max_close = seqs_to_dag(feasible_sequences)
        G_max_close2 = G_max_close.copy()
        
        G_min = nx.DiGraph()
        G_min.add_nodes_from(albp_problem["task_times"].keys())
        r_time = time.time()
        G_max_red_random, random_query_vals, random_bin_limit =  non_repeating_strat_w_eval(G_max_close2, G_min,albp_problem,n_queries, ex_fp )
        r_time = time.time()-r_time
        os = [d["OS"] for d in random_query_vals]
        val = [d["n_stations"] for d in random_query_vals]
        res_list.append({'method':'random', 'time': r_time, 'OS':os,'query_values':val,  'bin_lb':random_bin_limit, **metadata})
        # #hoffman
        mhh_res = do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_mhh_solve,selector_method=selector_method,seed=attempt_ind+base_seed)
        res_list.append({**metadata, **mhh_res, 'method':'hoffman'})
        #Prioriy
        priority_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_prioirity_solve,selector_method=selector_method,seed=attempt_ind+base_seed)
        res_list.append({**metadata, **priority_res, 'method':'priority'})
        # print("calculating ml results now")
        priority_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, myopic_ml_choice_edge,selector_method="ml", seed=attempt_ind+base_seed,ml_model=ml_model, )
        res_list.append({**metadata, **priority_res, 'method':'Adaboost'})
        if not fast_only:
            print("running vdls and bbr ")
            #vdls
            vdls_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_vdls_dict, selector_method=selector_method ,seed=attempt_ind+base_seed,time_limit=1)
            res_list.append({ **metadata, **vdls_res,'method':'vdls'})
            #bbr
            bbr_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_bbr_call, selector_method=selector_method,seed=attempt_ind+base_seed,time_limit=1, w_bin_pack=False)
            res_list.append({ **metadata, **bbr_res,'method':'bbr'})
        

    my_df = pd.DataFrame(res_list)
    my_df.to_csv(f"{save_folder}/{name}.csv", index=False)

def safe_constraint_elim(*args):
    try:
        return constraint_elim(*args)
    except Exception as e:
        print(f"Error in constraint_elim: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_func(x):
    print(f"Processing {x}")
    return x * 2

def main():
    parser = argparse.ArgumentParser(description="Run experiment with given dataset and parameters.")

    parser.add_argument(
        "--fp",
        type=str,
        required=True,
        help="Path to the dataset pickle file"
    )
    parser.add_argument(
        "--ex_fp",
        type=str,
        default='../BBR-for-SALBP1/SALB/SALB/salb',
        help="Path to the executable (default: %(default)s)"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        required=True,
        help="Output folder to store results"
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=4,
        help="Number of processes to use (default: %(default)s)"
    )
    parser.add_argument(
        "--n_xps",
        type=int,
        default=5,
        help="Number of experiments to run (default: %(default)s)"
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=10,
        help="Number of queries to remove a precedence constraint (default: %(default)s)"
    )
    parser.add_argument(
        "--ml_model_fp",
        type=str,
        required=False,
        default = "data/ml_models/trained_models/boost_edge_10_7.pkl",
        help="What ml weights to use (assuming use of Boost method). (default:%(default)s)"
    )
    parser.add_argument(
        "--dp_search",
        type=str,
        required=False,
        default = "myopic",
        help="When searching dp tree, what strategy to use. There is myopic, greedy,  (default:%(default)s)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        required=False,
        default = "all",
        help="what methods to use. (default:%(default)s)"
    )



    parser.add_argument(
        "--n_seq",
        type=int,
        default=1,
        help="Number of initial sequences that start the precedence graph (default: %(default)s)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of pkl dataset entry to start (default: %(default)s)"
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Index of pkl dataset entry to start (default: %(default)s)"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Seed of random number generator. Each try will increment the base seed by one(default: %(default)s)"
    )
    
    

    args = parser.parse_args()
    alb_dicts = open_salbp_pickle(args.fp)[args.start:args.end]
    if args.methods=="fast_only":
        fast_only= True
    else:
        fast_only=False
    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    #albp_problem = alb_dicts[0]

    print("starting seed: ", args.base_seed)
    #FOR TESTING 1 instance
    #constraint_elim(alb_dicts[1], 2, args.ex_fp, args.out_folder, args.n_queries,args.n_seq, 'myopic', boost_edge,base_seed=args.base_seed, fast_only=fast_only )
    #Parallelized run
    print(alb_dicts)
    with multiprocessing.Pool(args.pool_size) as pool:
        print("starting pool")
        _ = pool.starmap(safe_constraint_elim, [(alb,args.n_xps, args.ex_fp, args.out_folder, args.n_queries, args.n_seq, args.dp_search, args.ml_model_fp, args.base_seed, fast_only) for alb in alb_dicts])
        #results = pool.map(test_func, range(5))

    
if __name__ == "__main__":
    main()