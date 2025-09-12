import numpy as np
import networkx as nx
import random
import pandas as pd

import time


def get_all_positional_weight(G, key="task_time"):
    '''Gets the positional weight of the graph'''
    positional_weight = {}
    trans_G = nx.transitive_closure(G)
    #positional weight is the weight of the node plus the weight of its children
    for node in trans_G.nodes():
        positional_weight[node] = trans_G.nodes[node][key]
        for child in trans_G.neighbors(node):
            positional_weight[node] += trans_G.nodes[child][key]
    return positional_weight

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
            edge_neighbor_max_min[edge] = {"max": max(weights), "min": min(weights), "avg": sum(weights)/len(weights), "std": np.std(weights), "edge_0_weight": G.nodes[edge[0]][key], "edge_1_weight": G.nodes[edge[1]][key]}
        else:
            edge_neighbor_max_min[edge] = {"max": 0, "min": 0, "avg": 0, "std": 0, "edge_0_weight": G.nodes[edge[0]][key], "edge_1_weight": G.nodes[edge[1]][key]}
    return edge_neighbor_max_min

def longest_chain_containing_node(G, node):
    if node not in G:
        return 0
    # Step 2: Get longest path to nodes
    longest_to_node = get_stages(G, nx.topological_sort(G))

    # Step 2: Get longest path from `node`
    rev_G = nx.reverse(G, copy=True)
    longest_from_node = get_stages(rev_G, nx.topological_sort(rev_G))
    # Step 3: Compute total longest chain containing the node
    return longest_to_node[node] + longest_from_node[node] + 1  # +1 to include the node itself


def get_stages(G, topologically_sorted_nodes):
    longest_to_node = {n: 0 for n in G}  # Dictionary to store longest path to each node
    for n in topologically_sorted_nodes:  # Process in topological order
        for pred in G.predecessors(n):
            longest_to_node[n] = max(longest_to_node[n], longest_to_node[pred] + 1)
    return longest_to_node

def get_longest_chains_edges(G):
    # Step 1: Get longest paths to parents
    longest_to_node = get_stages(G, nx.topological_sort(G))

    # Step 2: Get longest paths from children
    rev_G = nx.reverse(G, copy=True)
    longest_from_node = get_stages(rev_G, nx.topological_sort(rev_G))
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


def get_edge_data(instance_name, alb):
    '''Gets the data for all of the edges for a given alb instance'''
    start_time = time.time()
    edge_list = []
    G = nx.DiGraph()
    G.add_nodes_from([str(i) for i in range(1, alb['num_tasks'] + 1)])
    G.add_edges_from(alb['precedence_relations'])
    #adds task times as node attributes
    nx.set_node_attributes(G, {i: {'task_time': alb['task_times'][str(i)]} for i in G.nodes})
    positional_weights = get_all_positional_weight(G)
    longest_chains_to, longest_chains_from = longest_weighted_chains(G)
    edge_weights = get_edge_neighbor_max_min_avg_std(G)
    walk_info = generate_walk_stats_ALB(G, num_walks=5, walk_length=10)

    for idx, edge in enumerate(alb['precedence_relations']):

        neighborhood_info = edge_weights[tuple(edge)]

        chain_info = get_longest_chain_for_edge( longest_chains_to, longest_chains_from,edge)
        chain_avg = np.mean(chain_info['weights'])
        chain_min = np.min(chain_info['weights'])
        chain_max = np.max(chain_info['weights'])
        chain_std = np.std(chain_info['weights'])
        parent_in_degree = G.in_degree(edge[0])
        parent_out_degree = G.out_degree(edge[0])

        parent_weight = neighborhood_info['edge_0_weight']
        parent_stage = longest_chains_to[edge[0]]
        parent_pos_weight = positional_weights[edge[0]]
        #gets the row of the parent's walk data
        parent_walk_data = walk_info[walk_info['node'] == edge[0]].copy()
        #drops node from parent_walk_data
        parent_walk_data.drop('node', axis=1, inplace=True)
        #converts parent_walk_data to a dictionary
        parent_walk_data = parent_walk_data.to_dict(orient='records')[0]
        
        child_weight = neighborhood_info['edge_1_weight']
        child_stage = longest_chains_to[edge[0]]
        child_in_degree = G.in_degree(edge[1])
        child_out_degree = G.out_degree(edge[1])
        #gets child walk data
        child_walk_data = walk_info[walk_info['node'] == edge[1]].copy()
        #drops node from parent_walk_data
        child_walk_data.drop('node', axis=1, inplace=True)
        #converts parent_walk_data to a dictionary
        child_walk_data = child_walk_data.to_dict(orient='records')[0]
        child_walk_data = {f'child_{key}': value for key, value in child_walk_data.items()}
        child_pos_weight = positional_weights[edge[1]]
        end_time = time.time() - start_time
        edge_list.append({'instance': instance_name, 
                            'edge': edge, 
                            'idx': idx, 
                            'parent_weight':parent_weight,
                            'parent_pos_weight': parent_pos_weight,
                            'parent_stage': parent_stage,
                            'child_weight':child_weight, 
                            'child_pos_weight': child_pos_weight, 
                            'child_stage': child_stage,
                            'stage_difference': parent_stage-child_stage,
                            'neighborhood_min': neighborhood_info['min'], 
                            'neighborhood_max': neighborhood_info['max'], 
                            'neighborhood_avg': neighborhood_info['avg'], 
                            'neighborhood_std': neighborhood_info['std'], 
                            'parent_in_degree': parent_in_degree, 
                            'parent_out_degree': parent_out_degree, 
                            'child_in_degree': child_in_degree, 
                            'child_out_degree': child_out_degree, 
                            'chain_avg': chain_avg, 
                            'chain_min': chain_min, 
                            'chain_max': chain_max, 
                            'chain_std': chain_std, 
                            'edge_data_time':end_time, 
                            **parent_walk_data, 
                            **child_walk_data})
    return edge_list


def randomized_kahns_algorithm(G, n_runs=10, weight_key='task_time'):
    ''' Runs n_runs of the randomized Kahns topological sort algorithm and returns them as a list. Naive implementation'''
    topological_sorts = []
    for _ in range(n_runs):
        top_sort = []
        weights = []
        G_copy = G.copy()
        start_weight = 0
        run_increasing = True
        n_runs = 0
        while G_copy.nodes():
            #gets all nodes with in degree 0
            zero_in_degree_nodes = [node for node in G_copy.nodes if G_copy.in_degree(node) == 0]
            if not zero_in_degree_nodes:
                raise ValueError('Graph has a cycle')
            #selects a random node with in degree 0
            node = random.choice(zero_in_degree_nodes)
            node_weight = G_copy.nodes[node][weight_key]
            if start_weight - node_weight < 0:
                if not run_increasing:
                    n_runs += 1
                    run_increasing = True
            elif start_weight - node_weight > 0:
                if run_increasing:
                    n_runs += 1
                    run_increasing = False
            start_weight = node_weight
            weights.append(node_weight)
            top_sort.append(node)
            G_copy.remove_node(node)
        topological_sorts.append({'sorted_nodes': top_sort, 'weights': weights, 'n_runs': n_runs})
    return topological_sorts

def random_walk_from_node(G, node, walk, walk_length, transition_probabilities, num_walks):
    """Perform a single random walk starting from one node."""
    if transition_probabilities[int(node)-1].sum() == 0 or np.isnan(transition_probabilities[int(node)-1]).all():
        # if the node has no outgoing edges, the walk will only contain the node
        return {
            'node': node,
            'walk': walk,
            'nodes_visited': [node] * num_walks,
            'task_times': [G.nodes[node]["task_time"]] * num_walks,
            'total_time': G.nodes[node]["task_time"] * num_walks,
            'min_time': G.nodes[node]["task_time"],
            'max_time': G.nodes[node]["task_time"],
            'n_unique_nodes': 1,
            'walk_length': walk_length
        }

    current_node = node         
    node_walks = [current_node]
    task_times = [G.nodes[current_node]["task_time"]]

    for step in range(walk_length):
        current_node = np.random.choice(G.nodes, p=transition_probabilities[int(current_node)-1])
        node_walks.append(current_node)
        task_times.append(G.nodes[current_node]["task_time"])

    return {
        'node': node,
        'walk': walk,
        'nodes_visited': node_walks,
        'task_times': task_times,
        'total_time': sum(task_times),
        'min_time': min(task_times),
        'max_time': max(task_times),
        'n_unique_nodes': len(set(node_walks)),
        'walk_length': walk_length
    }


def random_walks_ALB(G, num_walks, walk_length):
    """Performs random walks on each node. Note that the graph becomes undirected."""
    G = G.to_undirected()
    adj_matrix = nx.to_numpy_array(G)
    inv_row_sums = np.reciprocal(adj_matrix.sum(axis=1)).reshape(-1, 1)
    transition_probabilities = adj_matrix * inv_row_sums

    walks = []
    for node in G.nodes:
        for walk in range(num_walks):
            walks.append(random_walk_from_node(G, node, walk, walk_length, transition_probabilities, num_walks))

    return walks

# def random_walks_ALB(G, num_walks, walk_length):
#     '''performs a random walk on each node. Note that the graph becomes undirected'''
#     #transforms the graph into an undirected graph
#     G = G.to_undirected()
#     adj_matrix = nx.to_numpy_array(G)
#     inv_row_sums = np.reciprocal(adj_matrix.sum(axis=1)).reshape(-1, 1)
#     transition_probabilities = adj_matrix * inv_row_sums
#     walks = []
#     for node in G.nodes:
#         for walk in range(num_walks):
#             if transition_probabilities[int(node)-1].sum() == 0 or np.isnan(transition_probabilities[int(node)-1]).all():
#                 #if the node has no outgoing edges, the walk will only contain the node
#                 walks.append({'node':node , 'walk':walk, 'nodes_visited': [node]*num_walks, 'task_times':[G.nodes[node]["task_time"]]*num_walks,
#                                 'total_time':G.nodes[node]["task_time"]*num_walks, 'min_time':G.nodes[node]["task_time"], 
#                                 'max_time':G.nodes[node]["task_time"], 'n_unique_nodes':1, 'walk_length':walk_length})
#                 break
#             current_node = node         
#             node_walks = []
#             task_times = []
#             node_walks.append(current_node)
#             task_times.append(G.nodes[current_node]["task_time"])
#             for step in range(walk_length):
#                 current_node = np.random.choice(G.nodes, p=transition_probabilities[int(current_node)-1])
#                 node_walks.append(current_node)
#                 task_times.append(G.nodes[current_node]["task_time"])
#             walks.append({'node':node , 'walk':walk, 'nodes_visited': node_walks, 'task_times':task_times, 
#                           'total_time':sum(task_times), 'min_time':min(task_times), 'max_time':max(task_times), 
#                           'n_unique_nodes':len(set(node_walks)), 'walk_length':walk_length})
#     return walks

def generate_walk_stats_ALB(G, num_walks=5, walk_length=10):
    '''generates statistics for random walks performed on each node of graph G'''
    #Starts timer
    start_time = time.time()
    walks = random_walks_ALB(G, num_walks, walk_length)
    #For each node, combines the nodes_visited and task_times lists
    walk_df = pd.DataFrame(walks)
    walk_stats = walk_df.groupby('node').agg({'nodes_visited':'sum', 'task_times':'sum','total_time':'mean', 'min_time':'mean', 'max_time':'mean', 'n_unique_nodes':'mean', 'walk_length':'mean'}).reset_index()
    #renames columns
    walk_stats.columns = ['node', 'all_nodes_visited', 'all_task_times', 'rw_mean_total_time', 'rw_mean_min_time', 'rw_mean_max_time', 'rw_mean_n_unique_nodes', 'rw_mean_walk_length']
    walk_stats['rw_min'] = walk_stats['all_task_times'].apply(lambda x: min(x))
    walk_stats['rw_max'] = walk_stats['all_task_times'].apply(lambda x: max(x))
    walk_stats['rw_mean'] = walk_stats['all_task_times'].apply(lambda x: np.mean(x))
    walk_stats['rw_std'] = walk_stats['all_task_times'].apply(lambda x: np.std(x))
    walk_stats['rw_n_unique_nodes'] = walk_stats['all_nodes_visited'].apply(lambda x: len(set(x)))
    walk_stats['rw_elapsed_time'] = time.time() - start_time
    #drops all_nodes_visited and all_task_times columns
    walk_stats = walk_stats.drop(['all_nodes_visited', 'all_task_times'], axis=1)
    return walk_stats
