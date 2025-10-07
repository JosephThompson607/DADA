import networkx as nx
import numpy as np
import pandas as pd
import time
from metrics.node_and_edge_features import get_stages


def calculate_order_strength(dag):
    """
    Calculate the order strength of a directed acyclic graph (DAG).

    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    float
        The order strength of the DAG.
    """
    # Calculate the order strength of the DAG
    trans_closure = nx.transitive_closure(dag)
    #gets the num edges of the transitive closure
    num_edges = trans_closure.number_of_edges()
    #gets the number of nodes
    num_nodes = trans_closure.number_of_nodes()

    return 2 * num_edges / (num_nodes * (num_nodes - 1))


def average_number_of_immediate_predecessors(dag):
    """
    Calculate the average number of immediate predecessors (AIP) of the nodes in a directed acyclic graph (DAG).

    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    float
        The average number of immediate predecessors of the nodes in the DAG.
    """
    # Calculate the average number of immediate predecessors of the nodes in the DAG
    return dag.number_of_edges() / dag.number_of_nodes()

def maximum_degree(dag):
    """
    Calculate the maximum degree of a directed acyclic graph (DAG).

    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    int
        The maximum degree of the DAG.
    """
    # Calculate the maximum degree of the DAG
    max_in = max(dict(dag.in_degree()).values())
    max_out = max(dict(dag.out_degree()).values())

    #Gets the number of neighbors of each node

    max_neigh = max([dag.in_degree(node) + dag.out_degree(node) for node in dag.nodes()])
    return max_neigh, max_in, max_out

def degree_of_divergence(dag):
    """
    Calculate the degree of divergence of a directed acyclic graph (DAG).

    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    float
        The degree of divergence of the DAG.
    """
    # Calculate the degree of divergence of the DAG
    #gets the number of tasks without predecessors
    num_tasks_without_predecessors = len([node for node in dag.nodes() if dag.in_degree(node) == 0])
    if num_tasks_without_predecessors > 1:
        #ed is the number of edges plus the number of tasks without predecessors
        ed = dag.number_of_edges() + num_tasks_without_predecessors
    else:
        ed = dag.number_of_edges()
    div_degree = 1 - (dag.number_of_edges() + num_tasks_without_predecessors - dag.number_of_nodes()) / ed
    return div_degree

def degree_of_convergence(dag):
    """
    Calculate the degree of convergence of a directed acyclic graph (DAG).

    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    float
        The degree of convergence of the DAG.
    """
    # Calculate the degree of convergence of the DAG
    #gets the number of tasks without successors
    num_tasks_without_successors = len([node for node in dag.nodes() if dag.out_degree(node) == 0])
    if num_tasks_without_successors > 1:
        #ed is the number of edges plus the number of tasks without successors
        ec = dag.number_of_edges() + num_tasks_without_successors
    else:
        ec = dag.number_of_edges()
    conv_degree = 1 - (dag.number_of_edges() + num_tasks_without_successors - dag.number_of_nodes()) / ec
    return conv_degree

def get_n_suc_pred(dag):
    node_stats = {}
    
    for node in dag.nodes():
        pred_count  = 0
        succ_count = 0
        #counts the number of predecessors of each node
        node_pred = len([pred for pred in dag.predecessors(node)])
        node_succ = len([succ for succ in dag.successors(node)])
        node_stats[node] = (node_pred, node_succ)
    return node_stats
def bottleneck_stats(dag):
    """
    Detects bottleneck nodes and returns the number of them and their average degree. A bottleneck node is a node that is a successor of at least 2 nodes with no other successor
    and is a predecessor of at least 2 nodes with no other predecessor.

    Parameters
    ----------
    dag : nx.DiGraph

    
    Returns
    -------
    int n_bottlenecks 
        The number of bottleneck nodes in the DAG.
    float avg_degree
        The average degree of the bottleneck nodes.
    """
    bottlenecks = []
    #First we calculate the number of predecessors and successors of each node
    node_stats = get_n_suc_pred(dag)

    #Then we check if the node is a bottleneck
    for node in dag.nodes():
        candidate = True
        pred_count = 0
        for pred in dag.predecessors(node):
            if node_stats[pred][1] == 1:
                pred_count += 1
        if pred_count < 2:
            candidate = False
            continue
        succ_count = 0
        for succ in dag.successors(node):
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
        

def go_up_chain(node, dag):
    """
    Goes up the chain of a node in a directed acyclic graph (DAG). It stops when the predecessor of a node has more than one successor.
    Parameters
    ----------
    node : str
        The node in the DAG.
        dag : nx.DiGraph
        The directed acyclic graph.
        Returns
        -------
        list
        The chain of the node.
    """
    current_node = node
    chain = []
    while len(list(dag.predecessors(current_node))) == 1:
        pred = list(dag.predecessors(current_node))[0]
        if len(list(dag.successors(pred))) > 1 or len(list(dag.predecessors(pred))) > 1:
            break
        chain.append(pred)
        current_node = pred
    return chain

def go_down_chain(node, dag):
    """
    Goes down the chain of a node in a directed acyclic graph (DAG). It stops when the successor of a node has more than one predecessor.
    Parameters
    ----------
    node : str
        The node in the DAG.
        dag : nx.DiGraph
        The directed acyclic graph.
        Returns
        -------
        list
        The chain of the node.
    """
    current_node = node
    chain = []
    while len(list(dag.successors(current_node))) == 1:
        succ = list(dag.successors(current_node))[0]
        if len(list(dag.successors(succ))) > 1 or len(list(dag.predecessors(succ))) > 1:
            break
        chain.append(succ)
        current_node = succ
    return chain

def chain_stats(dag):
    """
    Detects chain nodes and returns the number of them and their average length,
    a chain node is a node that is a predecessor of only one node and a successor of only one node. Chains only count if they are longer than two nodes.
    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    int n_chains
        The number of chain nodes in the DAG.
    float avg_length
        The average length of the chains.
    """
    node_stats = get_n_suc_pred(dag)
    chains = []
    #only considers nodes that have at most one predecessor and one successor
    
    nodes_to_consider = [node for node in dag.nodes() if node_stats[node][0] <= 1 and node_stats[node][1] <= 1]
    while len(nodes_to_consider) > 0:
        node = nodes_to_consider.pop()
        chain = go_up_chain(node, dag)
        chain.append(node)
        chain += go_down_chain(node, dag)
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

        

def get_n_isolated_nodes(dag):
    """
    Get the number of isolated nodes in a directed acyclic graph (DAG).
    
    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns 
    -------
    int
        The number of isolated nodes in the DAG.
    """
    return len([node for node in dag.nodes() if dag.in_degree(node) == 0 and dag.out_degree(node) == 0])




def get_n_tasks_without_predecessors(dag):
    """
    Get the number of tasks without predecessors in a directed acyclic graph (DAG).
    
    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns 
    -------
    int
        The number of tasks without predecessors in the DAG.
    """
    return len([node for node in dag.nodes() if dag.in_degree(node) == 0])

def make_alb_digraph(SALBP_instance):
    G = nx.DiGraph()
    for node, task_time in SALBP_instance['task_times'].items():
        G.add_node(node, task_time=task_time)

    G.add_edges_from(SALBP_instance["precedence_relations"])
    return G

def get_graph_metrics(SALBP_instance):
    G = make_alb_digraph(SALBP_instance)
    return generate_graph_metrics(G)


def get_driscoll_stats(dag):
    n_stages = get_n_stages(dag)
    prec_strength = (n_stages-1) / (dag.number_of_nodes()-1)
    node_stages = get_stages(dag, nx.topological_sort(dag))
    average_stage = sum(node_stages.values())/dag.number_of_nodes()
    prec_bias = average_stage/n_stages 
    prec_index = (prec_strength + prec_bias)/2
    return n_stages, prec_strength, prec_bias, prec_index


def generate_graph_metrics(dag):
    """
    Generate the graph metrics of a directed acyclic graph (DAG)."""
    start_time = time.time()
    # Calculate the order strength of the DAG
    order_strength = calculate_order_strength(dag)

    # Calculate the average number of immediate predecessors of the nodes in the DAG
    aip = average_number_of_immediate_predecessors(dag)

    # Calculate the maximum degree of the DAG
    max_degree, max_in_degree, max_out_degree = maximum_degree(dag)

    # Calculate the degree of divergence of the DAG
    divergence_degree = degree_of_divergence(dag)

    # Calculate the degree of convergence of the DAG
    convergence_degree = degree_of_convergence(dag)

    # Calculate the number of bottleneck nodes and their average degree
    n_bottlenecks, avg_degree_bot = bottleneck_stats(dag)

    # Calculate the number of chain nodes and their average length
    n_chains, avg_length, nodes_in_chains = chain_stats(dag)
    #Number of isolated nodes
    n_isolated_nodes = get_n_isolated_nodes(dag)
    #Precedence strength, precedence bias, and precedence_index
    n_stages, precedence_strength, precedence_bias, precedence_index = get_driscoll_stats(dag)
    end_time = time.time() - start_time
    res_dict = {
        'n_edges': dag.number_of_edges(),
        'order_strength': order_strength,
        'average_number_of_immediate_predecessors': aip,
        'max_degree': max_degree,
        'max_in_degree': max_in_degree,
        'max_out_degree': max_out_degree,
        'divergence_degree': divergence_degree,
        'convergence_degree': convergence_degree,
        'n_bottlenecks': n_bottlenecks,
        'share_of_bottlenecks': n_bottlenecks / dag.number_of_nodes(),
        'avg_degree_of_bottlenecks': avg_degree_bot,
        'n_chains': n_chains,
        'avg_chain_length': avg_length,
        'nodes_in_chains': nodes_in_chains,
        'n_stages': n_stages,
        'stages_div_n': n_stages/dag.number_of_edges(),
        'prec_strength': precedence_strength,
        'prec_bias': precedence_bias,
        'prec_index': precedence_index,
        'n_isolated_nodes': n_isolated_nodes,
        'share_of_isolated_nodes': n_isolated_nodes / dag.number_of_nodes(),
        'n_tasks_without_predecessors': get_n_tasks_without_predecessors(dag),
        'share_of_tasks_without_predecessors': get_n_tasks_without_predecessors(dag) / dag.number_of_nodes(),
        'avg_tasks_per_stage': dag.number_of_nodes() / get_n_stages(dag),
        'graph_feature_time': end_time
    }
    return res_dict


def get_n_stages(dag):
    """
    Get the number of stages of a directed acyclic graph (DAG).

    Parameters
    ----------
    dag : nx.DiGraph
        The directed acyclic graph.

    Returns
    -------
    int
        The number of stages of the DAG.
    """
    # Get the number of stages of the DAG
    return nx.dag_longest_path_length(dag) + 1