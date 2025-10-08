import re
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import subprocess
import sys
import os
build_dir = '/home/jot240/DADA/SALBP_ILS/build/'
sys.path.insert(0, build_dir)
sys.path.append('src/')
from metrics.graph_features import calculate_order_strength
from metrics.node_and_edge_features import randomized_kahns_algorithm
from alb_instance_compressor import *
from constraint_discovery import *
from SALBP_solve import salbp1_bbr_call, parse_alb_results_new_bbr

import yaml
from copy import deepcopy
import ILS_ALBP

def sequence_then_search(orig_salbp, n_xps, res_folder_fp, ex_fp, depth_limit, q_check_tl=10, base_seed=42, n_sorts=2):
    for xp in range(n_xps):
        seed = base_seed + xp
        name = str(orig_salbp['name']).split('/')[-1].split('.')[0] 
        topo_sorts= albp_to_sequences(orig_salbp,n_sorts, seed=seed)
        seq_metadata = {'seed':base_seed, 'n_sorts':n_sorts}
        feasible_sequences = []
        for i in topo_sorts:
            feasible_sequences.append(i['sorted_nodes'])
        print(feasible_sequences)

        G_max_close = seqs_to_dag(feasible_sequences)
        folder_name = f'{res_folder_fp}/{name}_sorts_{n_sorts}_seed_{seed}/'
        out_folder = Path(folder_name)
        out_folder.mkdir(parents=True, exist_ok=True)
        out_fp = folder_name + f'res_{name}_{seed}.csv'

        problem_data_fp = folder_name + f'{name}_seed_{seed}_metadata.yaml'

        results, problem_data = enumerate_removals_bfs(G_max_close, orig_salbp, ex_fp, out_fp, problem_data_fp, depth_limit, q_check_tl, seq_metadata)



def enumerate_removals_bfs(G_max_close, orig_salbp,ex_fp, out_fp,problem_data_fp, depth_limit=None, q_check_tl=10, G_max_metadata={}):
    """
    Enumerate all possible sequences of edge removals in a BFS fashion,
    recomputing the transitive reduction after each removal.

    Parameters
    ----------
    prediction_graph : nx.DiGraph
        The starting DAG.
    depth_limit : int or None
        Maximum number of removals in a sequence. If None, explore fully.

    Returns
    -------
    results : list[{'depth':int, edges:list, 'n_stations':int, 'name':string, 'bin_lb':int, 'max_depth':int, 'eval_time':int}]
        A set of unique sequences, where each sequence is a frozenset of removed edges.
    """
    name = str(orig_salbp['name']).split('/')[-1].split('.')[0]
    orig_solver = False
    if "BBR-for-SALBP1/" in ex_fp:
        orig_solver = True
    baseline = salbp1_bbr_call(orig_salbp,ex_fp, 1, time_limit=q_check_tl, orig_bbr=orig_solver)
    bin_lb = baseline['bin_lb'] #same no matter what, can use it here
    s_true = baseline['n_stations']
    #for metadata_about the graph
    G_max_red = nx.transitive_reduction(G_max_close)
    G_max_red_edges = list(G_max_red.edges())
    start_OS = calculate_order_strength(G_max_close)
    G_true = albp_to_nx(orig_salbp)
    true_OS = calculate_order_strength(G_true)
    G_min = nx.DiGraph()
    G_min.add_nodes_from(orig_salbp["task_times"].keys())
    problem_data = {**orig_salbp, 'G_max_edges':G_max_red_edges, 'G_max_OS': start_OS, **G_max_metadata }
    problem_data['name']=name
    queue = [(G_max_close.copy(), [])]  # Queue of (transitive_closure_graph, removed_edges) tuples
    history = set()
       

    n_queries = 0
    with open(problem_data_fp, 'w') as file:
        yaml.safe_dump(problem_data, file, 
                default_flow_style=False
            )

    while queue:
        G, removed_edges = queue.pop(0)
        
        #This is to avoid repeat computations (i.e. e1->e2, e2->e1)
        edge_set = frozenset(removed_edges)
        if edge_set in history:
            continue
        # Store the current removal sequence
        history.add(edge_set)
        
        depth = len(removed_edges)
        current_OS = calculate_order_strength(G)
        red = nx.transitive_reduction(G)
        test_salbp, new_to_old = set_new_edges(red, orig_salbp)
        for key in new_to_old.keys():
            print(f"new value : {key}, old value {new_to_old[key]}, new task time {test_salbp['task_times'][key]} old task time {orig_salbp['task_times'][new_to_old[key]]}")
        eval_time = time.time()
        res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=q_check_tl, w_bin_pack=False, orig_bbr=orig_solver)
        print('res task assignments: ', res)
        real_task_assignment = [0] * len(res['task_assignments'])
        for i in range( len(res['task_assignments'])):
            correct_task = int(new_to_old[str(i+1)] ) -1
            real_task_assignment[correct_task] = res['task_assignments'][i]
        print('corrected task assignments', real_task_assignment)
        eval_time = time.time()-eval_time

        res_dict = {'name':name, 
                        'depth':depth, 
                        'edges':list(edge_set), 
                        'n_stations':res['n_stations'], 
                        'optimal': res['verified_optimality'],
                        'true_n_stations':baseline['n_stations'],
                        'bin_lb':bin_lb, 
                        'depth_limit':depth_limit, 
                        'n_queries': n_queries,
                        'eval_time':eval_time,
                        'start_OS': start_OS,
                        'true_OS': true_OS,
                        'current_OS': current_OS,
                        'task_assignment': real_task_assignment
                        }
        results_df = pd.DataFrame([res_dict])

        results_df.to_csv(out_fp, mode='a', index=False, header=not os.path.exists(out_fp))
        df = pd.read_csv(out_fp)

        print(df.iloc[-1].to_dict())  # prints the last row as a dictionary
        #check to see if it reached objective value
        if res['n_stations'] <= s_true: 
            continue
       
        # Check if we've reached the depth limit
        elif depth_limit is not None and depth >= depth_limit:
            continue 
        # Iterate through the valid edges
        edges = get_possible_edges(red, G_min)
        #print("These are the edges", edges)

        for edge in edges:
            new_G= G.copy()
            new_G, _, G_min, successful_removal = focused_query_prec_set(new_G, red, G_min, G_true,edge)

            n_queries = n_queries +  1
            if successful_removal:
                # Create new state for the next level of the BFS
                new_removed_edges = removed_edges + [edge]
                # Add the new state to the queue
                queue.append((new_G, new_removed_edges))




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
        default='../bbr-salbp/bbr',
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
        "--depth_limit",
        type=int,
        default=100,
        help="Number of queries to remove a precedence constraint (default: %(default)s)"
    )
    # parser.add_argument(
    #     "--dp_search",
    #     type=str,
    #     required=False,
    #     default = "greedy",
    #     help="When searching dp tree, what strategy to use. (default:%(default)s)"
    # )


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
    

    args = parser.parse_args()
    alb_dicts = open_salbp_pickle(args.fp)[args.start:args.end]
    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    #albp_problem = alb_dicts[0]
    with multiprocessing.Pool(args.pool_size) as pool:
        results = pool.starmap(sequence_then_search, [(alb, args.n_xps, args.out_folder, args.ex_fp,  args.depth_limit, 20, 42, args.n_seq) for alb in alb_dicts])
        

    
if __name__ == "__main__":
    main()