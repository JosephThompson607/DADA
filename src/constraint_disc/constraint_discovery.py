import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import sys

import time
import xgboost as xgb
import copy
import yaml
from ml_search import *
from set_new_edges import *
sys.path.append('src')
from beam_search import beam_search_mh, beam_search_ml
from metrics.graph_features import calculate_order_strength
from datetime import date

from metrics.node_and_edge_features import randomized_kahns_algorithm

from alb_instance_compressor import *
from SALBP_solve import salbp1_bbr_call, salbp1_prioirity_solve, salbp1_vdls_dict, salbp1_hoff_solve, load_and_backup_configs
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



def give_probabilities(false_edges, true_edges, true_prob=0.1, false_prob=0.9, xi=0.2, seed=None, precision=2):
    if seed is not None:
        random.seed(seed)
    
    true_probs = {edge: true_prob for edge in true_edges}
    false_probs = {edge: false_prob for edge in false_edges}
    probs = {**true_probs, **false_probs}

    n_perturb = int(len(probs) * xi)
    edges_to_perturb = random.sample(list(probs.keys()), n_perturb)

    for edge in edges_to_perturb:
        probs[edge] = 1 - probs[edge]
    probs = {edge: round(prob, precision) for edge, prob in probs.items()}
    return probs

def edge_prob_generation(alb_instance, G_max_close, true_prob=0.1, false_prob=0.9, xi=0.2,seed=None):
    false_edges, true_edges = get_false_edges(alb_instance, G_max_close.edges())
    edge_probs = give_probabilities(false_edges, true_edges, true_prob=true_prob, false_prob=false_prob,xi=xi,seed=seed)
    return edge_probs

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
        #copying over edge data
        G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
    elif G_max_red.has_edge(edge[0], edge[1]) and G_true.has_edge(edge[0], edge[1]):
        G_min.add_edge(edge[0], edge[1])
    else:
        print(f"edge {edge} not in transitive reduction")
        print(list(G_max_red.edges()))
    return G_max_close, G_max_red, G_min, successful_removal










# def myopic_choice_mh(edges, orig_salbp, G_max_red_orig, mh,  **mhkwargs):
#     G_max_red = G_max_red_orig.copy()
#     best_edge = edges[0]
#     remaining_edges =edges[1:]
#     G_max_red.clear_edges()
#     G_max_red.add_edges_from(remaining_edges)
#     new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
#     #res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=1, w_bin_pack=False)
#     res = mh(new_salbp, **mhkwargs)
#     station_best = res['n_stations']
#     best_edge = edges[0]

#     for i, edge in enumerate(edges[1:]):
#         # Create list without current item
#         remaining_edges = edges[:i] + edges[i+1:]
#         G_max_red.clear_edges()
#         G_max_red.add_edges_from(remaining_edges)
#         new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
#         #res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=1, w_bin_pack=False)
#         res = mh(new_salbp, **mhkwargs)
#         if res['n_stations'] < station_best:
#             station_best = res['n_stations']
#             best_edge = edge
#     print("selecting", best_edge)
#     return best_edge, station_best      





def make_random_weight_generator(seed=None, output_name="n_stations"):
    rng = random.Random(seed)
    def random_weight(*_, **__):
        return {output_name: rng.random()}
    return random_weight

def constant_weight(new_salbp,*args,output_name = "n_stations", **kwargs):
    return {output_name:1}

def os_weight(new_salbp,*args,output_name = "n_stations", **kwargs):
    G = albp_to_nx(new_salbp)
    os = calculate_order_strength(G)
    return {output_name:os}

def random_valid(G_max_red, G_min,  rng):
    edges = get_possible_edges(G_max_red, G_min)
    return rng.choice(edges)


def best_first_reduction(G_max_close, G_min,  orig_salbp, n_queries , ex_fp, mh, q_check_tl=3, selector_method='beam_mh',ml_model = None, seed=42,**mhkwargs):
    G_true = albp_to_nx(orig_salbp)
    start = time.time()
    G_max_red = nx.transitive_reduction(G_max_close)
    order_strength = calculate_order_strength(G_max_red)
    test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    orig_bbr = False
    if "BBR-for-SALBP1/" in ex_fp:
        orig_bbr = True
    res = salbp1_bbr_call(test_salbp,ex_fp, 1, w_bin_pack=True, time_limit=q_check_tl, orig_bbr=orig_bbr)
    new_kwargs = {**mhkwargs, 'ex_fp':ex_fp, 'orig_bbr':orig_bbr, 'seed':seed}
    orig_n_stations = res['n_stations']
    n_stations = orig_n_stations
    bin_limit = res['bin_lb']
    random.seed(seed)
    if selector_method =='random':
        rng = random.Random(seed)
    #print(f"SALBP number of stations before : {orig_n_stations}, ellapsed time {time.time()- start}, bin lb {bin_limit} OS {order_strength}")
    query_vals = [{"n_stations":orig_n_stations, "OS":order_strength, "q_time": time.time()-start, "edge":()}]
    for q in range(n_queries): 
        if n_stations == bin_limit:
            print(f"reached bpp lower bound terminating after {q} queries")
            break
        
        
        q_start = time.time()

           
        if selector_method == 'ml':
            edge, _ = beam_search_ml( orig_salbp, G_max_close,G_min, ml_model , **new_kwargs)
            #edge, probability = best_first_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model, **new_kwargs )
        elif selector_method == 'beam_mh':
            edge, _ = beam_search_mh( orig_salbp, G_max_close,G_min, mh, init_sol=res, **new_kwargs)
        elif selector_method =='random':
            edge = random_valid( G_max_red,G_min,rng)
        
        G_max_close, G_max_red, G_min, _ = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
        order_strength = calculate_order_strength(G_max_red)
        test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=q_check_tl, orig_bbr=orig_bbr)
        n_stations = res['n_stations']
        q_time = time.time()-q_start
        query_vals.append({"n_stations":n_stations, "OS":order_strength, "q_time": q_time, "edge":edge})
    return G_max_red, query_vals, bin_limit


def get_feasible_seq(albp, n_seq, seed=42):
    topo_sorts= albp_to_sequences(albp, n_seq, seed=seed)
    feasible_sequences = []
    for i in topo_sorts:
        feasible_sequences.append(i['sorted_nodes'])
    return feasible_sequences


def do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, mh, selector_method, time_limit=1,seed=42, **mh_kwargs):
    '''Runs a single xp and returns the results'''
    G_max_close2 = G_max_close.copy()
          
    G_min = nx.DiGraph()
    G_min.add_nodes_from(albp_problem["task_times"].keys())
    r_time = time.time()
    _, query_vals, bin_limit =  best_first_reduction(G_max_close2, G_min, albp_problem, n_queries, ex_fp, mh, selector_method=selector_method, time_limit=time_limit,seed=seed, **mh_kwargs)
    r_time = time.time()-r_time
    os = [d["OS"] for d in query_vals]
    val = [d["n_stations"] for d in query_vals]
    q_times = [d["q_time"] for d in query_vals]
    edges = [d["edge"] for d in query_vals]
    return { 'time': r_time,  'OS':os,'query_values':val, 'q_time':q_times,  'bin_lb':bin_limit, 'edges':edges}



def constraint_elim(albp_problem, mh_methods, n_tries, ex_fp, save_folder, n_queries,n_start_sequences, xp_config_fp,beam_config, base_seed=None,  q_check_tl=3, edge_prob_data={}):
    name = str(albp_problem['name']).split('/')[-1].split('.')[0]            
    
    xp_config, ml_config, ml_model = load_and_backup_configs(xp_config_fp, backup_folder=save_folder)
        
    res_list = []
    for attempt_ind in range(n_tries):
        #Sets a seed so that the different methods have the same seed for each attempt for comparability
        trial_seed = attempt_ind+base_seed
        #albp_problem = parse_alb("/Users/letshopethisworks2/Documents/phd_paper_material/DADA/test.alb")
        feasible_sequences = get_feasible_seq(albp_problem,n_start_sequences, seed=trial_seed)
        metadata = {'dataset':albp_problem['name'], 'name': name, 'n_queries':n_queries, 'attempt':attempt_ind, 'n_start_sequences':n_start_sequences,  "beam_width": beam_config['width'], "beam_depth":beam_config['depth']}
        G_max_close = seqs_to_dag(feasible_sequences)
        if edge_prob_data:
            metadata = {**metadata, **edge_prob_data}
            edge_probs = edge_prob_generation(albp_problem, G_max_close, xi=edge_prob_data['xi'], true_prob=edge_prob_data['true_prob'], false_prob=edge_prob_data['false_prob'],seed=trial_seed)
            nx.set_edge_attributes(G_max_close, edge_probs, name='prob')

        print("solving ",albp_problem['name'])
        if any(method in mh_methods for method in ["random", "all", "fast"]):
            print("running random")     
            random_weight = make_random_weight_generator(seed=trial_seed)
            rand_res = do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, random_weight,selector_method='random',seed=trial_seed, q_check_tl=q_check_tl, beam_config={"width":1, "depth":1}, )
            res_list.append({**metadata, **rand_res, 'method':'random'})

        if any(method in mh_methods for method in ["hoffman", "all", "fast"]):   
            print("running hoffman")     
            # #hoffman
            mhh_res = do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_hoff_solve,selector_method='beam_mh',seed=trial_seed, q_check_tl=q_check_tl, beam_config=beam_config, **xp_config['hoff'])
            res_list.append({**metadata, **mhh_res, 'method':'hoffman'})
        if any(method in mh_methods for method in ["priority", "all", "fast"]):  
            print("running priority")
            #Prioriy
            priority_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_prioirity_solve,selector_method='beam_mh',seed=trial_seed, q_check_tl=q_check_tl, beam_config=beam_config, **xp_config['priority'])
            res_list.append({**metadata, **priority_res, 'method':'priority'})
        # print("calculating ml results now")
        if any(method in mh_methods for method in ["ml", "all", "fast"]):  
            priority_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, best_first_ml_choice_edge,selector_method="ml", seed=trial_seed,ml_model=ml_model, q_check_tl=q_check_tl, beam_config=beam_config, ml_config = ml_config)
            res_list.append({**metadata, **priority_res, 'method':'xgboost'})
        if any(method in mh_methods for method in ["probability", "all", "fast"]):  
                print("running probability")
                #Prioriy
                priority_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, constant_weight,selector_method='beam_mh',seed=trial_seed, q_check_tl=q_check_tl, beam_config=beam_config)
                res_list.append({**metadata, **priority_res, 'method':'probability'})
        # print("calculating ml results now")
        
        #vdls
        if any(method in mh_methods for method in ["vdls", "all", ]):  
            print("running vdls")
            vdls_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_vdls_dict, selector_method='beam_mh',seed=trial_seed,time_limit=1, q_check_tl=q_check_tl)
            res_list.append({ **metadata, **vdls_res,'method':'vdls'})
        #bbr
        if any(method in mh_methods for method in ["bbr", "all", ]):  
            print("running bbr")
            bbr_res= do_greedy_run(albp_problem, n_queries, G_max_close, ex_fp, salbp1_bbr_call, selector_method='beam_mh',seed=trial_seed,time_limit=1, w_bin_pack=False, q_check_tl=q_check_tl)
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
        "--xp_config_fp",
        type=str,
        required=False,
        default="data/xp_config/hoff_simple.yaml",
        help="config file for xp (default: %(default)s)"
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
        "--depth",
        type=int,
        default=1,
        help="For beam search, how many layers to solve. Default is 2 (default: %(default)s)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3,
        help="For beam search, how many beams. Default is 3 (default: %(default)s)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        required=False,
        default = "fast",
        help="what methods to use. (default:%(default)s)"
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=None,
        help="Edge probability generation. Proportation of edges to randomly swap to simulate noise in the probability estimates(default: %(default)s)"
    )
    parser.add_argument(
        "--false_prob",
        type=float,
        default=0.9,
        help="Edge probability generation. Initial probability given to edges that are not part of the real graph that they are not part of the graph(default: %(default)s)"
    )
    parser.add_argument(
        "--true_prob",
        type=float,
        default=0.1,
        help="Edge probability generation. Initial probability given to edges that are part of the real graph that they are not part of the graph(default: %(default)s)"
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
    parser.add_argument(
        "--q_check_tl",
        type=int,
        default=5,
        help="How long to spend on BBR evaluation of previous constraint elimination (default: %(default)s)"
    )

    
    


    args = parser.parse_args()
    alb_dicts = open_salbp_pickle(args.fp)[args.start:args.end]
    out_folder = Path(args.out_folder + date.today().isoformat())
    out_folder.mkdir(parents=True, exist_ok=True)
    #albp_problem = alb_dicts[0]
    beam_config = {"width":args.width, "depth":args.depth}
    if args.xi:
        edge_prob_data = {"xi":args.xi, "true_prob":args.true_prob, "false_prob":args.false_prob}
    else:
        edge_prob_data = {}
    print("starting seed: ", args.base_seed)
    #FOR TESTING 1 instance
    #constraint_elim(alb_dicts[1], 2, args.ex_fp, args.out_folder, args.n_queries,args.n_seq, 'best_first', boost_edge,base_seed=args.base_seed, fast_only=fast_only )
    #Parallelized run
    with multiprocessing.Pool(args.pool_size) as pool:
        _ = pool.starmap(constraint_elim, [(alb,
                                            args.methods,
                                            args.n_xps, 
                                            args.ex_fp, 
                                            out_folder, 
                                            args.n_queries, 
                                            args.n_seq, 
                                            args.xp_config_fp,
                                            beam_config, 
                                            args.base_seed,  
                                            args.q_check_tl,
                                            edge_prob_data) for alb in alb_dicts])
        #results = pool.map(test_func, range(5))

    
if __name__ == "__main__":
    main()