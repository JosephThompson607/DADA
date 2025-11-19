import networkx as nx
import numpy as np
import pandas as pd
import time
from metrics.node_and_edge_features import get_stages


def calculate_order_strength(G_max_red, G_max_close=None):
    """
    Calculate the order strength of a directed acyclic graph (G_max_red).

    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    float
        The order strength of the G_max_red.
    """
    if not G_max_close:
        print("CACULATING TRANSITIVE CLOSURE")
        # Calculate the order strength of the G_max_red
        trans_closure = nx.transitive_closure(G_max_red)
    else:
        trans_closure = G_max_close
    #gets the num edges of the transitive closure
    num_edges = trans_closure.number_of_edges()
    #gets the number of nodes
    num_nodes = trans_closure.number_of_nodes()

    return 2 * num_edges / (num_nodes * (num_nodes - 1))


def average_number_of_immediate_predecessors(G_max_red):
    """
    Calculate the average number of immediate predecessors (AIP) of the nodes in a directed acyclic graph (G_max_red).

    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    float
        The average number of immediate predecessors of the nodes in the G_max_red.
    """
    # Calculate the average number of immediate predecessors of the nodes in the G_max_red
    return G_max_red.number_of_edges() / G_max_red.number_of_nodes()

def maximum_degree(G_max_red):
    """
    Calculate the maximum degree of a directed acyclic graph (G_max_red).

    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    int
        The maximum degree of the G_max_red.
    """
    # Calculate the maximum degree of the G_max_red
    max_in = max(dict(G_max_red.in_degree()).values())
    max_out = max(dict(G_max_red.out_degree()).values())

    #Gets the number of neighbors of each node

    max_neigh = max([G_max_red.in_degree(node) + G_max_red.out_degree(node) for node in G_max_red.nodes()])
    return max_neigh, max_in, max_out

def degree_of_divergence(G_max_red):
    """
    Calculate the degree of divergence of a directed acyclic graph (G_max_red).

    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    float
        The degree of divergence of the G_max_red.
    """
    # Calculate the degree of divergence of the G_max_red
    #gets the number of tasks without predecessors
    num_tasks_without_predecessors = len([node for node in G_max_red.nodes() if G_max_red.in_degree(node) == 0])
    if num_tasks_without_predecessors > 1:
        #ed is the number of edges plus the number of tasks without predecessors
        ed = G_max_red.number_of_edges() + num_tasks_without_predecessors
    else:
        ed = G_max_red.number_of_edges()
    div_degree = 1 - (G_max_red.number_of_edges() + num_tasks_without_predecessors - G_max_red.number_of_nodes()) / ed
    return div_degree

def degree_of_convergence(G_max_red):
    """
    Calculate the degree of convergence of a directed acyclic graph (G_max_red).

    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    float
        The degree of convergence of the G_max_red.
    """
    # Calculate the degree of convergence of the G_max_red
    #gets the number of tasks without successors
    num_tasks_without_successors = len([node for node in G_max_red.nodes() if G_max_red.out_degree(node) == 0])
    if num_tasks_without_successors > 1:
        #ed is the number of edges plus the number of tasks without successors
        ec = G_max_red.number_of_edges() + num_tasks_without_successors
    else:
        ec = G_max_red.number_of_edges()
    conv_degree = 1 - (G_max_red.number_of_edges() + num_tasks_without_successors - G_max_red.number_of_nodes()) / ec
    return conv_degree

def get_n_suc_pred(G_max_red):
    node_stats = {}
    
    for node in G_max_red.nodes():
        pred_count  = 0
        succ_count = 0
        #counts the number of predecessors of each node
        node_pred = len([pred for pred in G_max_red.predecessors(node)])
        node_succ = len([succ for succ in G_max_red.successors(node)])
        node_stats[node] = (node_pred, node_succ)
    return node_stats
def bottleneck_stats(G_max_red):
    """
    Detects bottleneck nodes and returns the number of them and their average degree. A bottleneck node is a node that is a successor of at least 2 nodes with no other successor
    and is a predecessor of at least 2 nodes with no other predecessor.

    Parameters
    ----------
    G_max_red : nx.DiGraph

    
    Returns
    -------
    int n_bottlenecks 
        The number of bottleneck nodes in the G_max_red.
    float avg_degree
        The average degree of the bottleneck nodes.
    """
    bottlenecks = []
    #First we calculate the number of predecessors and successors of each node
    node_stats = get_n_suc_pred(G_max_red)

    #Then we check if the node is a bottleneck
    for node in G_max_red.nodes():
        candidate = True
        pred_count = 0
        for pred in G_max_red.predecessors(node):
            if node_stats[pred][1] == 1:
                pred_count += 1
        if pred_count < 2:
            candidate = False
            continue
        succ_count = 0
        for succ in G_max_red.successors(node):
            if node_stats[succ][0] == 1:
                succ_count += 1
        if succ_count < 2:
            candidate = False
        if candidate:
            bottlenecks.append((node, node_stats[node][0] + node_stats[node][1]))
    n_bottlenecks = len(bottlenecks)
    if n_bottlenecks == 0:
        return 0, 0
    avg_degree = np.mean([bottleneck[1] for bottleneck in bottlenecks])
    return n_bottlenecks, avg_degree
        

def go_up_chain(node, G_max_red):
    """
    Goes up the chain of a node in a directed acyclic graph (G_max_red). It stops when the predecessor of a node has more than one successor.
    Parameters
    ----------
    node : str
        The node in the G_max_red.
        G_max_red : nx.DiGraph
        The directed acyclic graph.
        Returns
        -------
        list
        The chain of the node.
    """
    current_node = node
    chain = []
    while len(list(G_max_red.predecessors(current_node))) == 1:
        pred = list(G_max_red.predecessors(current_node))[0]
        if len(list(G_max_red.successors(pred))) > 1 or len(list(G_max_red.predecessors(pred))) > 1:
            break
        chain.append(pred)
        current_node = pred
    return chain

def go_down_chain(node, G_max_red):
    """
    Goes down the chain of a node in a directed acyclic graph (G_max_red). It stops when the successor of a node has more than one predecessor.
    Parameters
    ----------
    node : str
        The node in the G_max_red.
        G_max_red : nx.DiGraph
        The directed acyclic graph.
        Returns
        -------
        list
        The chain of the node.
    """
    current_node = node
    chain = []
    while len(list(G_max_red.successors(current_node))) == 1:
        succ = list(G_max_red.successors(current_node))[0]
        if len(list(G_max_red.successors(succ))) > 1 or len(list(G_max_red.predecessors(succ))) > 1:
            break
        chain.append(succ)
        current_node = succ
    return chain

def chain_stats(G_max_red):
    """
    Detects chain nodes and returns the number of them and their average length,
    a chain node is a node that is a predecessor of only one node and a successor of only one node. Chains only count if they are longer than two nodes.
    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    int n_chains
        The number of chain nodes in the G_max_red.
    float avg_length
        The average length of the chains.
    """
    node_stats = get_n_suc_pred(G_max_red)
    chains = []
    #only considers nodes that have at most one predecessor and one successor
    
    nodes_to_consider = [node for node in G_max_red.nodes() if node_stats[node][0] <= 1 and node_stats[node][1] <= 1]
    while len(nodes_to_consider) > 0:
        node = nodes_to_consider.pop()
        chain = go_up_chain(node, G_max_red)
        chain.append(node)
        chain += go_down_chain(node, G_max_red)
        if len(chain) >= 2:
            chains.append(chain)
        #removes the nodes that are part of the chain
        nodes_to_consider = [node for node in nodes_to_consider if node not in chain]
    n_chains = len(chains)
    chain_lengths = [len(chain) for chain in chains]
    if n_chains == 0:
        return 0, 0, 0
    avg_length = np.mean(chain_lengths)
    nodes_in_chains = sum(chain_lengths)
    return n_chains, avg_length, nodes_in_chains

        

def get_n_isolated_nodes(G_max_red):
    """
    Get the number of isolated nodes in a directed acyclic graph (G_max_red).
    
    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns 
    -------
    int
        The number of isolated nodes in the G_max_red.
    """
    return len([node for node in G_max_red.nodes() if G_max_red.in_degree(node) == 0 and G_max_red.out_degree(node) == 0])




def get_n_tasks_without_predecessors(G_max_red):
    """
    Get the number of tasks without predecessors in a directed acyclic graph (G_max_red).
    
    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns 
    -------
    int
        The number of tasks without predecessors in the G_max_red.
    """
    return len([node for node in G_max_red.nodes() if G_max_red.in_degree(node) == 0])

def make_alb_digraph(SALBP_instance):
    G = nx.DiGraph()
    for node, task_time in SALBP_instance['task_times'].items():
        G.add_node(node, task_time=task_time)

    G.add_edges_from(SALBP_instance["precedence_relations"])
    return G



def get_graph_metrics(SALBP_instance,G_max_close=None,G_max_red=None):
    if not G_max_red:
        G = make_alb_digraph(SALBP_instance)
    else:
        G = G_max_red
    return generate_graph_metrics( G, G_max_close)


def get_driscoll_stats(G_max_red):
    n_stages = get_n_stages(G_max_red)
    prec_strength = (n_stages-1) / (G_max_red.number_of_nodes()-1)
    node_stages = get_stages(G_max_red, nx.topological_sort(G_max_red))
    average_stage = sum(node_stages.values())/G_max_red.number_of_nodes()
    prec_bias = average_stage/n_stages 
    prec_index = (prec_strength + prec_bias)/2
    return n_stages, prec_strength, prec_bias, prec_index


# def generate_graph_metrics(G_max_red):
#     """
#     Generate the graph metrics of a directed acyclic graph (G_max_red)."""
#     start_time = time.time()
#     # Calculate the order strength of the G_max_red
#     order_strength = calculate_order_strength(G_max_red)

#     # Calculate the average number of immediate predecessors of the nodes in the G_max_red
#     aip = average_number_of_immediate_predecessors(G_max_red)

#     # Calculate the maximum degree of the G_max_red
#     max_degree, max_in_degree, max_out_degree = maximum_degree(G_max_red)

#     # Calculate the degree of divergence of the G_max_red
#     divergence_degree = degree_of_divergence(G_max_red)

#     # Calculate the degree of convergence of the G_max_red
#     convergence_degree = degree_of_convergence(G_max_red)

#     # Calculate the number of bottleneck nodes and their average degree
#     n_bottlenecks, avg_degree_bot = bottleneck_stats(G_max_red)

#     # Calculate the number of chain nodes and their average length
#     n_chains, avg_length, nodes_in_chains = chain_stats(G_max_red)
#     #Number of isolated nodes
#     n_isolated_nodes = get_n_isolated_nodes(G_max_red)
#     #Precedence strength, precedence bias, and precedence_index
#     n_stages, precedence_strength, precedence_bias, precedence_index = get_driscoll_stats(G_max_red)
#     end_time = time.time() - start_time
#     res_dict = {
#         'n_edges': G_max_red.number_of_edges(),
#         'order_strength': order_strength,
#         'average_number_of_immediate_predecessors': aip,
#         'max_degree': max_degree,
#         'max_in_degree': max_in_degree,
#         'max_out_degree': max_out_degree,
#         'divergence_degree': divergence_degree,
#         'convergence_degree': convergence_degree,
#         'n_bottlenecks': n_bottlenecks,
#         'share_of_bottlenecks': n_bottlenecks / G_max_red.number_of_nodes(),
#         'avg_degree_of_bottlenecks': avg_degree_bot,
#         'n_chains': n_chains,
#         'avg_chain_length': avg_length,
#         'nodes_in_chains': nodes_in_chains,
#         'n_stages': n_stages,
#         'stages_div_n': n_stages/G_max_red.number_of_edges(),
#         'prec_strength': precedence_strength,
#         'prec_bias': precedence_bias,
#         'prec_index': precedence_index,
#         'n_isolated_nodes': n_isolated_nodes,
#         'share_of_isolated_nodes': n_isolated_nodes / G_max_red.number_of_nodes(),
#         'n_tasks_without_predecessors': get_n_tasks_without_predecessors(G_max_red),
#         'share_of_tasks_without_predecessors': get_n_tasks_without_predecessors(G_max_red) / G_max_red.number_of_nodes(),
#         'avg_tasks_per_stage': G_max_red.number_of_nodes() / get_n_stages(G_max_red),
#         'graph_feature_time': end_time
#     }
#     return res_dict

def generate_graph_metrics(G_max_red, G_max_close=None):
    """
    Generate the graph metrics of a directed acyclic graph (G_max_red)."""
    t_func_start = time.perf_counter()
    start_time = time.time()
    
    profile_stats = {
        'calculate_order_strength': 0.0,
        'average_number_of_immediate_predecessors': 0.0,
        'maximum_degree': 0.0,
        'degree_of_divergence': 0.0,
        'degree_of_convergence': 0.0,
        'bottleneck_stats': 0.0,
        'chain_stats': 0.0,
        'get_n_isolated_nodes': 0.0,
        'get_driscoll_stats': 0.0,
        'get_n_tasks_without_predecessors': 0.0,
        'get_n_stages': 0.0,
        'dict_construction': 0.0,
        'function_total': 0.0
    }
    
    # Calculate the order strength of the G_max_red
    t_start = time.perf_counter()
    order_strength = calculate_order_strength(G_max_red, G_max_close)
    profile_stats['calculate_order_strength'] = time.perf_counter() - t_start
    
    # Calculate the average number of immediate predecessors of the nodes in the G_max_red
    t_start = time.perf_counter()
    aip = average_number_of_immediate_predecessors(G_max_red)
    profile_stats['average_number_of_immediate_predecessors'] = time.perf_counter() - t_start
    
    # Calculate the maximum degree of the G_max_red
    t_start = time.perf_counter()
    max_degree, max_in_degree, max_out_degree = maximum_degree(G_max_red)
    profile_stats['maximum_degree'] = time.perf_counter() - t_start
    
    # Calculate the degree of divergence of the G_max_red
    t_start = time.perf_counter()
    divergence_degree = degree_of_divergence(G_max_red)
    profile_stats['degree_of_divergence'] = time.perf_counter() - t_start
    
    # Calculate the degree of convergence of the G_max_red
    t_start = time.perf_counter()
    convergence_degree = degree_of_convergence(G_max_red)
    profile_stats['degree_of_convergence'] = time.perf_counter() - t_start
    
    # Calculate the number of bottleneck nodes and their average degree
    t_start = time.perf_counter()
    n_bottlenecks, avg_degree_bot = bottleneck_stats(G_max_red)
    profile_stats['bottleneck_stats'] = time.perf_counter() - t_start
    
    # Calculate the number of chain nodes and their average length
    t_start = time.perf_counter()
    n_chains, avg_length, nodes_in_chains = chain_stats(G_max_red)
    profile_stats['chain_stats'] = time.perf_counter() - t_start
    
    # Number of isolated nodes
    t_start = time.perf_counter()
    n_isolated_nodes = get_n_isolated_nodes(G_max_red)
    profile_stats['get_n_isolated_nodes'] = time.perf_counter() - t_start
    
    # Precedence strength, precedence bias, and precedence_index
    t_start = time.perf_counter()
    n_stages, precedence_strength, precedence_bias, precedence_index = get_driscoll_stats(G_max_red)
    profile_stats['get_driscoll_stats'] = time.perf_counter() - t_start
    
    end_time = time.time() - start_time
    
    # Additional calls for dict construction
    t_start = time.perf_counter()
    n_tasks_without_pred = get_n_tasks_without_predecessors(G_max_red)
    profile_stats['get_n_tasks_without_predecessors'] = time.perf_counter() - t_start
    
    t_start = time.perf_counter()
    n_stages_calc = get_n_stages(G_max_red)
    profile_stats['get_n_stages'] = time.perf_counter() - t_start
    
    # Dictionary construction
    t_start = time.perf_counter()
    res_dict = {
        'n_edges': G_max_red.number_of_edges(),
        'order_strength': order_strength,
        'average_number_of_immediate_predecessors': aip,
        'max_degree': max_degree,
        'max_in_degree': max_in_degree,
        'max_out_degree': max_out_degree,
        'divergence_degree': divergence_degree,
        'convergence_degree': convergence_degree,
        'n_bottlenecks': n_bottlenecks,
        'share_of_bottlenecks': n_bottlenecks / G_max_red.number_of_nodes(),
        'avg_degree_of_bottlenecks': avg_degree_bot,
        'n_chains': n_chains,
        'avg_chain_length': avg_length,
        'nodes_in_chains': nodes_in_chains,
        'n_stages': n_stages,
        'stages_div_n': n_stages / G_max_red.number_of_edges(),
        'prec_strength': precedence_strength,
        'prec_bias': precedence_bias,
        'prec_index': precedence_index,
        'n_isolated_nodes': n_isolated_nodes,
        'share_of_isolated_nodes': n_isolated_nodes / G_max_red.number_of_nodes(),
        'n_tasks_without_predecessors': n_tasks_without_pred,
        'share_of_tasks_without_predecessors': n_tasks_without_pred / G_max_red.number_of_nodes(),
        'avg_tasks_per_stage': G_max_red.number_of_nodes() / n_stages_calc,
        'graph_feature_time': end_time
    }
    profile_stats['dict_construction'] = time.perf_counter() - t_start
    
    profile_stats['function_total'] = time.perf_counter() - t_func_start
    
    # Print profile summary
    # print("\n=== PROFILE: generate_graph_metrics ===")
    # print(f"Total time: {profile_stats['function_total']:.4f}s")
    # print(f"Graph size: {G_max_red.number_of_nodes()} nodes, {G_max_red.number_of_edges()} edges")
    # print(f"\nMetric calculations:")
    # print(f"  calculate_order_strength:              {profile_stats['calculate_order_strength']:.4f}s ({profile_stats['calculate_order_strength']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  average_number_of_immediate_predecessors: {profile_stats['average_number_of_immediate_predecessors']:.4f}s ({profile_stats['average_number_of_immediate_predecessors']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  maximum_degree:                        {profile_stats['maximum_degree']:.4f}s ({profile_stats['maximum_degree']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  degree_of_divergence:                  {profile_stats['degree_of_divergence']:.4f}s ({profile_stats['degree_of_divergence']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  degree_of_convergence:                 {profile_stats['degree_of_convergence']:.4f}s ({profile_stats['degree_of_convergence']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  bottleneck_stats:                      {profile_stats['bottleneck_stats']:.4f}s ({profile_stats['bottleneck_stats']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  chain_stats:                           {profile_stats['chain_stats']:.4f}s ({profile_stats['chain_stats']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  get_n_isolated_nodes:                  {profile_stats['get_n_isolated_nodes']:.4f}s ({profile_stats['get_n_isolated_nodes']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  get_driscoll_stats:                    {profile_stats['get_driscoll_stats']:.4f}s ({profile_stats['get_driscoll_stats']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  get_n_tasks_without_predecessors:      {profile_stats['get_n_tasks_without_predecessors']:.4f}s ({profile_stats['get_n_tasks_without_predecessors']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  get_n_stages:                          {profile_stats['get_n_stages']:.4f}s ({profile_stats['get_n_stages']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  dict_construction:                     {profile_stats['dict_construction']:.4f}s ({profile_stats['dict_construction']/profile_stats['function_total']*100:.1f}%)")
    # print("=" * 39)
    
    return res_dict

def get_n_stages(G_max_red):
    """
    Get the number of stages of a directed acyclic graph (G_max_red).

    Parameters
    ----------
    G_max_red : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    int
        The number of stages of the G_max_red.
    """
    # Get the number of stages of the G_max_red
    return nx.dag_longest_path_length(G_max_red) + 1