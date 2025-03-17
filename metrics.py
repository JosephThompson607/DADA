import numpy as np
import networkx as nx



def get_all_reverse_positional_weight(G, key="task_time"):
    '''Gets the reverse positional weight of the graph'''
    rev_G = G.reverse()
    rpw = get_all_positional_weight(rev_G, key=key)
    return rpw

def get_all_children(G):
    '''Gets all the children of the nodes in the  graph'''
    children_dict = {}
    for node in G.nodes():
        children_dict[node] = list(G.successors(node))
    return children_dict

def get_all_parents(G):
    '''Gets all the parents of the nodes in the graph'''
    parents_dict = {}
    for node in G.nodes():
        parents_dict[node] = list(G.predecessors(node))
    return parents_dict

def get_all_successors(G):
    '''Gets all the succesors of the nodes in the graph'''
    trans_G = nx.transitive_closure(G)
    succesors_dict = {}
    for node in trans_G.nodes():
        succesors_dict[node] = list(G.predecessors(node))
    return succesors_dict

def get_all_predecessors(G):
    '''Gets all the predecessors of the nodes in the graph'''
    trans_G = nx.transitive_closure(G)
    predecessors_dict = {}
    for node in trans_G.nodes():
        predecessors_dict[node] = list(G.predecessors(node))
    return predecessors_dict


def get_edge_neighbor_max_min_avg_std(G, key="task_time"):
    '''For each edge, gets the maximum and minimum weight of its neighbors'''
    edge_neighbor_max_min = {}
    for edge in G.edges():
        #gets the weights of the predecessors of the first node in the edge
        pred_weights = [G.nodes[pred][key] for pred in G.predecessors(edge[0])] 
        #gets the weights of the successors of the second node in the edge
        succ_weights = [G.nodes[succ][key] for succ in G.successors(edge[1])] 
        #adds the max and min of the weights to the edge_neighbor_max_min dictionary
        weights = pred_weights + succ_weights
        if weights:
            edge_neighbor_max_min[edge] = {"max": max(weights), "min": min(weights), "avg": sum(weights)/len(weights), "std": np.std(weights)}
        else:
            edge_neighbor_max_min[edge] = {"max": 0, "min": 0, "avg": 0, "std": 0}
    return edge_neighbor_max_min

def longest_chain_containing_node(G, node):
    if node not in G:
        return 0

    # Step 1: Get longest path to `node`
    longest_to_node = {n: 0 for n in G}  # Dictionary to store longest path to each node
    for n in nx.topological_sort(G):  # Process in topological order
        for pred in G.predecessors(n):
            longest_to_node[n] = max(longest_to_node[n], longest_to_node[pred] + 1)

    # Step 2: Get longest path from `node`
    longest_from_node = {n: 0 for n in G}  # Dictionary to store longest path from each node
    for n in reversed(list(nx.topological_sort(G))):  # Process in reverse topological order
        for succ in G.successors(n):
            longest_from_node[n] = max(longest_from_node[n], longest_from_node[succ] + 1)

    # Step 3: Compute total longest chain containing the node
    return longest_to_node[node] + longest_from_node[node] + 1  # +1 to include the node itself


def get_longest_chains_edges(G):
    # Step 1: Get longest path to `node`
    longest_to_node = {n: 0 for n in G}  # Dictionary to store longest path to each node
    for n in nx.topological_sort(G):  # Process in topological order
        for pred in G.predecessors(n):
            longest_to_node[n] = max(longest_to_node[n], longest_to_node[pred] + 1)

    # Step 2: Get longest path from `node`
    longest_from_node = {n: 0 for n in G}  # Dictionary to store longest path from each node
    for n in reversed(list(nx.topological_sort(G))):  # Process in reverse topological order
        for succ in G.successors(n):
            longest_from_node[n] = max(longest_from_node[n], longest_from_node[succ] + 1)
    edge_list = []
    # Step 3: Compute total longest chain containing the edge
    for edge in G.edges():
        edge_list.append((edge, longest_to_node[edge[0]] + longest_from_node[edge[1]] + 1))
    return edge_list


def longest_weighted_chains(G):
    # Step 1: Get longest path to `node`
    longest_chains_to=  {str(n[0]):{'nodes': [n[0]], 'weights': [n[1]['task_time']]} for n in G.nodes(data=True)}
    #print("these are the longest chains to", longest_chains_to)
    for n in nx.topological_sort(G):  # Process in topological order
        for pred in G.predecessors(n):
            # print("this is n", n, "this is pred", pred)
            # print(longest_chains_to[pred]['nodes'])
            # print(longest_chains_to[n]['nodes'])
            if len(longest_chains_to[pred]['nodes']) + 1 > len(longest_chains_to[n]['nodes']):
                longest_chains_to[n]['nodes'] = longest_chains_to[pred]['nodes'] + [n]
                node_weight = G.nodes[n]['task_time']
                longest_chains_to[n]['weights'] = longest_chains_to[pred]['weights'] + [node_weight]
                
    # Step 2: Get longest path from `node`
    longest_chains_from = {str(n[0]):{'nodes': [n[0]], 'weights': [n[1]['task_time']]} for n in G.nodes(data=True)}
    for n in reversed(list(nx.topological_sort(G))):  # Process in reverse topological order
        for suc in G.successors(n):
            if len(longest_chains_from[suc]['nodes']) + 1 > len(longest_chains_from[n]['nodes']):
                longest_chains_from[n]['nodes'] = longest_chains_from[suc]['nodes'] + [n]
                node_weight = G.nodes[n]['task_time']
                longest_chains_from[n]['weights'] = longest_chains_from[suc]['weights'] + [node_weight]

    return longest_chains_to, longest_chains_from

def get_longest_chain_for_edge(longest_chains_to, longest_chains_from, edge):
    parent_chain = longest_chains_to[edge[0]]
    child_chain = longest_chains_from[edge[1]]
    return {'nodes': parent_chain['nodes'] + child_chain['nodes'][::-1], 'weights': parent_chain['weights'] + child_chain['weights'][::-1]}

