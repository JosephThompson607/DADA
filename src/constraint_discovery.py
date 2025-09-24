import re
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import subprocess
import sys
sys.path.append('../src')
from metrics.node_and_edge_features import randomized_kahns_algorithm
from alb_instance_compressor import *
from SALBP_solve import salbp1_bbr_call
build_dir = '/Users/letshopethisworks2/CLionProjects/SALBP_ILS/cmake-build-python_interface/'
sys.path.insert(0, build_dir)
build_dir_2 = '/home/jot240/DADA/SALBP_ILS/build/'
sys.path.insert(0, build_dir_2)
import ILS_ALBP


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
    print("from dict", SALBP_dict["precedence_relations"])

    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    #prints the edges from the graph
    print("from graph", G.edges())
    nx.draw(G,pos, with_labels = True)
    plt.show()

def plot_salbp_graph(G):

    P = nx.nx_pydot.to_pydot(G)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    #prints the edges from the graph
    print("from graph", G.edges())
    nx.draw(G,pos, with_labels = True)
    plt.show()

def albp_to_sequences(albp, n_sequences, ):
    G = albp_to_nx(albp)
    return randomized_kahns_algorithm(G, n_runs=n_sequences)

def seqs_to_dag(orig_sequences):
    feasible_sequences = orig_sequences.copy()
    G = nx.DiGraph()
    #Add nodes
    seq1 = feasible_sequences.pop(0)
    n = len(seq1)
    G.add_nodes_from(seq1)
    #Initial graph
    print(seq1)
    for i in range(1, n+1):
        print("i", str(i))
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
    test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=1)
    orig_n_stations = res['value']
    n_stations = orig_n_stations
    bin_limit = res['bin_lb']
    print("res: ", res)
    print(f"SALBP number of stations before : {orig_n_stations}")
    query_vals = []
    for q in range(n_queries):
        if n_stations == bin_limit:
            print(f"reached bpp lower bound terminating after {q} queries")
            break
        edges = get_possible_edges(G_max_red, G_min)
        edge =  random.choice(edges)
        G_max_close, G_max_red, G_min = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
        test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=1)
        n_stations = res['value']
        print(f"SALBP number of stations at query  {q} : {n_stations}")
        query_vals.append(n_stations)

    return G_max_red, bin_limit, query_vals



def greedy_choice(edges, orig_salbp, G_max_red_orig, ex_fp ):
    G_max_red = G_max_red_orig.copy()
    station_best = 100000000
    for i, edge in enumerate(edges):
        # Create list without current item
        remaining_edges = edges[:i] + edges[i+1:]
        G_max_red.clear_edges()
        G_max_red.add_edges_from(remaining_edges)
        new_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(new_salbp,ex_fp, 1, time_limit=1)
        if res['value'] < station_best:
            station_best = res['value']
            print("best res ", res)
            edge = edge
            if res['value'] == res['bin_lb']:
                break
    return edge, station_best


  
        




def greedy_reduction(G_max_close, G_min,  orig_salbp, n_queries , ex_fp,):
    G_true = albp_to_nx(orig_salbp)
    
    G_max_red = nx.transitive_reduction(G_max_close)
    test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
    res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=1)
    orig_n_stations = res['value']
    n_stations = orig_n_stations
    bin_limit = res['bin_lb']
    print("res: ", res)
    print(f"SALBP number of stations before : {orig_n_stations}")
    query_vals = []
    for q in range(n_queries): 
        if n_stations == bin_limit:
            print(f"reached bpp lower bound terminating after {q} queries")
            break
        
        edges = get_possible_edges(G_max_red, G_min)
        random.shuffle(edges)
        edge, n_stations= greedy_choice(edges, orig_salbp, G_max_red, ex_fp)
        print("checking edge: ", edge, " with value: ",  orig_n_stations-n_stations)
        G_max_close, G_max_red, G_min = focused_query_prec_set(G_max_close, G_max_red, G_min, G_true,edge)
        test_salbp, new_to_old = set_new_edges(G_max_red, orig_salbp)
        res = salbp1_bbr_call(test_salbp,ex_fp, 1, time_limit=1)
        n_stations = res['value']
        print(f"SALBP number of stations at query  {q+1} : {n_stations}")
        query_vals.append(n_stations)
    return G_max_red, query_vals, bin_limit

