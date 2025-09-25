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
from metrics.graph_features import calculate_order_strength
sys.path.append('src')
from metrics.node_and_edge_features import randomized_kahns_algorithm
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
    if G_max_red.has_edge(edge[0], edge[1]) and not G_true.has_edge(edge[0], edge[1]):
        G_max_close.remove_edge(edge[0], edge[1])

        G_max_red= nx.transitive_reduction(G_max_close)
    elif G_max_red.has_edge(edge[0], edge[1]) and G_true.has_edge(edge[0], edge[1]):
        G_min.add_edge(edge[0], edge[1])
    else:
        print(f"edge {edge} not in transitive reduction")
    return G_max_close, G_max_red, G_min




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
        G_max_close, G_max_red, G_min = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
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
    new_salbp = orig_salbp.copy()
    new_task_times = {}
    for task, time in orig_salbp['task_times'].items():
        new_task_times[old_to_new[task]] = time
    new_salbp['precedence_relations'] = [[old_to_new[e1], old_to_new[e2]] for e1,e2 in G_max_red.edges()]
    new_salbp['task_times'] = new_task_times
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
    print(f"SALBP number of stations before : {orig_n_stations}, order strength: {order_strength}")
    query_vals = [{"n_stations":orig_n_stations, "OS":order_strength}]
    for q in range(n_queries):
        if n_stations == bin_limit:
            print(f"reached bpp lower bound terminating after {q} queries")
            break
        edges = get_possible_edges(G_max_red, G_min)

        edge =  random.choice(edges)
        G_max_close, G_max_red, G_min = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
        order_strength = calculate_order_strength(G_max_red)
        test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=3, w_bin_pack=False)
        n_stations = res['n_stations']

        print(f"SALBP number of stations at query  {q} : {n_stations}, number of edges: {len(edges)}, order strength {order_strength}")
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
    print("initial best: ", station_best, " edge ", best_edge)
    for i, edge in enumerate(edges[1:]):
        if res['n_stations'] < station_best:
            station_best = res['n_stations']
            best_edge = edge
            print("current best: ", station_best, " edge ", best_edge)
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
    res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=1, w_bin_pack=False)
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


def greedy_choice_mh(edges, orig_salbp, G_max_red_orig, mh, **mhkwargs ):
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
            return best_edge, station_best     
        # Create list without current item
        remaining_edges = edges[:i] + edges[i+1:]
        G_max_red.clear_edges()
        G_max_red.add_edges_from(remaining_edges)
        new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        #res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=1, w_bin_pack=False)
        res = mh(new_salbp, **mhkwargs)
    return best_edge, station_best      




def greedy_reduction(G_max_close, G_min,  orig_salbp, n_queries , ex_fp, mh, q_check_tl=3, **mhkwargs):
    G_true = albp_to_nx(orig_salbp)
    start = time.time()
    G_max_red = nx.transitive_reduction(G_max_close)
    order_strength = calculate_order_strength(G_max_red)
    test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=q_check_tl)
    init_sol = res['task_assignments']
    new_kwargs = {**mhkwargs, 'initial_solution':init_sol, 'ex_fp':ex_fp}
    orig_n_stations = res['n_stations']
    n_stations = orig_n_stations
    bin_limit = res['bin_lb']
    print(f"SALBP number of stations before : {orig_n_stations}, ellapsed time {time.time()- start}, bin lb {bin_limit} OS {order_strength}")
    query_vals = [{"n_stations":orig_n_stations, "OS":order_strength}]
    for q in range(n_queries): 
        if n_stations == bin_limit:
            print(f"reached bpp lower bound terminating after {q} queries")
            break
        
        edges = get_possible_edges(G_max_red, G_min)
        random.shuffle(edges)
        greedy_time = time.time()
        edge, n_stations= greedy_choice_mh(edges, orig_salbp, G_max_red, mh, **new_kwargs)
        #print("checking edge: ", edge, " with value: ",  orig_n_stations-n_stations, " ellapsed time ", time.time()-start, " greedy time ", time.time()-greedy_time)
        G_max_close, G_max_red, G_min = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
        order_strength = calculate_order_strength(G_max_red)
        test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=q_check_tl)
        n_stations = res['n_stations']
        print(f"SALBP number of stations at query  {q+1} : {n_stations}, order strength: {order_strength}")
        query_vals.append({"n_stations":n_stations, "OS":order_strength})
    return G_max_red, query_vals, bin_limit


def get_feasible_seq(albp, n_seq, seed=42):
    topo_sorts= albp_to_sequences(albp, n_seq, seed=seed)
    feasible_sequences = []
    for i in topo_sorts:
        feasible_sequences.append(i['sorted_nodes'])
    return feasible_sequences


def do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, mh, time_limit=1, **mh_kwargs):
    G_max_close2 = G_max_close.copy()
          
    G_min = nx.DiGraph()
    G_min.add_nodes_from(albp_problem["task_times"].keys())
    r_time = time.time()
    G_max_red_random, random_query_vals, random_bin_limit =  greedy_reduction(G_max_close2, G_min, albp_problem, n_queries, ex_fp, mh, time_limit=time_limit, **mh_kwargs)
    print(random_query_vals)
    r_time = time.time()-r_time
    os = [d["OS"] for d in random_query_vals]
    val = [d["n_stations"] for d in random_query_vals]
    return { 'time': r_time,  'OS':os,'query_values':val,  'bin_lb':random_bin_limit}



def constraint_elim(albp_problem, n_tries, ex_fp, save_folder, n_queries,n_start_sequences):
    name = str(albp_problem['name']).split('/')[-1].split('.')[0]            

    res_list = []
    for attempt_ind in range(n_tries):
            
            #albp_problem = parse_alb("/Users/letshopethisworks2/Documents/phd_paper_material/DADA/test.alb")
            feasible_sequences = get_feasible_seq(albp_problem,n_start_sequences, seed=attempt_ind)
            metadata = {'dataset':albp_problem['name'], 'name': name, 'n_queries':n_queries, 'attempt':attempt_ind, 'n_start_sequences':n_start_sequences}
            G_max_close = seqs_to_dag(feasible_sequences)
            G_max_close2 = G_max_close.copy()
          
            G_min = nx.DiGraph()
            G_min.add_nodes_from(albp_problem["task_times"].keys())
            r_time = time.time()
            G_max_red_random, random_query_vals, random_bin_limit =  non_repeating_strat_w_eval(G_max_close2, G_min,albp_problem,n_queries, ex_fp )
            print(random_query_vals)
            r_time = time.time()-r_time
            os = [d["OS"] for d in random_query_vals]
            val = [d["n_stations"] for d in random_query_vals]
            res_list.append({'method':'random', 'time': r_time, 'OS':os,'query_values':val,  'bin_lb':random_bin_limit, **metadata})
            # #hoffman
            mhh_res = do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_mhh_solve)
            res_list.append({**metadata, **mhh_res, 'method':'hoffman'})
            #Prioriy
            priority_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_prioirity_solve)
            res_list.append({**metadata, **priority_res, 'method':'priority'})
            #vdls
            vdls_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_vdls_dict, time_limit=1)
            res_list.append({ **metadata, **vdls_res,'method':'vdls'})
            #bbr
            bbr_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_bbr_call, time_limit=1, w_bin_pack=False)
            res_list.append({ **metadata, **bbr_res,'method':'bbr'})

    my_df = pd.DataFrame(res_list)
    my_df.to_csv(f"{save_folder}/{name}.csv", index=False)

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
        "--n_queries",
        type=int,
        default=10,
        help="Number of queries to remove a precedence constraint (default: %(default)s)"
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
    

    args = parser.parse_args()
    alb_dicts = open_salbp_pickle(args.fp)[args.start:args.end]
    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    #albp_problem = alb_dicts[0]
    with multiprocessing.Pool(args.pool_size) as pool:
        results = pool.starmap(constraint_elim, [(alb,args.n_xps, args.ex_fp, args.out_folder, args.n_queries, args.n_seq) for alb in alb_dicts])
        

    
if __name__ == "__main__":
    main()