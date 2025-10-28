import networkx as nx
import copy
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


def get_possible_edges(G_max_red, G_min, ban_list = []):
    G_min_close = nx.transitive_closure(G_min)
    candidates = []
    for u,v,data in  G_max_red.edges(data=True):
        if not G_min_close.has_edge(u, v) and (u,v) not in ban_list:
            if data:
                candidates.append((u, v, data['prob']))
            else:
                candidates.append((u, v))
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



