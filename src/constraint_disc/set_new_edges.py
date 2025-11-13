import networkx as nx
import copy

def sorted_copy(G):
    """Because networkx uses dictionaries, order can get messed up of edges in transitive reduction, this enforces order"""
    H = G.__class__()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(sorted(G.edges(data=True)))
    return H


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
        topo_order = list(nx.topological_sort(sorted_copy(G)))
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


def get_possible_edges(G_max_red, G_min, remaining_budget=1e6,ban_list = []):
    G_min_close = nx.transitive_closure(G_min)
    candidates = []
    for u,v,data in  G_max_red.edges(data=True):
        time = data['t_cost']
        if not G_min_close.has_edge(u, v) and (u,v) not in ban_list and remaining_budget-time>=0 :
            candidates.append((u, v, data['prob'], time))

    return candidates

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

def get_false_edges(orig_salb, G_max_edges):
    true_edges = set([(e[0], e[1]) for e in orig_salb['precedence_relations']])
    G_max_edges = set([(e[0], e[1]) for e in G_max_edges])
    false_edges = G_max_edges - true_edges
    return false_edges, true_edges


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
        G_max_red= sorted_copy(nx.transitive_reduction(G_max_close))
        #copying over edge data
        G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
    elif G_max_red.has_edge(edge[0], edge[1]) and G_true.has_edge(edge[0], edge[1]):
        G_min.add_edge(edge[0], edge[1])
    else:
        print(f"edge {edge} not in transitive reduction")
        print(list(G_max_red.edges()))
    return G_max_close, G_max_red, G_min, successful_removal



def uncertain_query_prec_set(G_max_close, G_max_red, G_min, edge, rng):
    """This function only removes an edge if it is part of the transitive reduction of 
        Ē but not in the real precedence constraints, and also if it succeds a random trial based on its current probability"""
    
    prob = edge[2]
    
    successful_removal = False
    if G_max_red.has_edge(edge[0], edge[1]) and rng.random()< prob:
        G_max_close.remove_edge(edge[0], edge[1])
        successful_removal = True
        G_max_red= nx.transitive_reduction(G_max_close)
        #copying over edge data
        G_max_red.add_edges_from((u, v, G_max_close.edges[u, v]) for u, v in G_max_red.edges)
    elif G_max_red.has_edge(edge[0], edge[1]):
        G_min.add_edge(edge[0], edge[1])
    else:
        print(f"edge {edge} not in transitive reduction")
        print(list(G_max_red.edges()))
    return G_max_close, G_max_red, G_min, successful_removal





