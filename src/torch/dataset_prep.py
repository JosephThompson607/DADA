import glob
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import ast
from multiprocessing import Pool

# Assuming 'src' is in the same directory as your notebook
src_path = os.path.abspath("src/")
if src_path not in sys.path:
    sys.path.append(src_path)
src_path = os.path.abspath("src/torch")
if src_path not in sys.path:
    sys.path.append(src_path)
from alb_instance_compressor import *
from copy import deepcopy
from metrics.node_and_edge_features import *
from metrics.graph_features import *
from metrics.time_metrics import *
from salb_dataset import EdgeClassificationDataset
from data_prep import *
import multiprocessing


from torch_geometric.data import Data, Dataset
import torch
import pandas as pd
import pickle

def get_node_and_graph_feat(instance_df, node_level_features, graph_level_features, debug_time = False):
        #Getting node feat
        df = instance_df[node_level_features + graph_level_features + ["instance","edge"]].copy()
        # Step 1: Extract parent and child IDs from edge column
        df["parent"] = df["edge"].apply(lambda e: str(e[0]) if isinstance(e, (list, tuple)) else str(eval(e)[0]))
        df["child"] = df["edge"].apply(lambda e: str(e[1]) if isinstance(e, (list, tuple)) else str(eval(e)[1]))

        # Step 2: Separate feature columns
        df = df.rename(columns={col: f'parent_{col}' for col in df.columns if col.startswith('rw')})
        feature_cols = [c for c in df.columns if c not in ["edge", "parent", "child","instance"]]
        
        # Step 3: Identify which columns belong to parent/child/both
        parent_cols = [c for c in feature_cols if "parent" in c ]
        child_cols = [c for c in feature_cols if "child" in c] 
        orphan_cols = [c for c in feature_cols if c not in parent_cols + child_cols+ graph_level_features]
        if len(orphan_cols) > 0:
            print('Warning: some columns were not attributed to a node')
            print('here are the features that do not match', orphan_cols)
        print('building parent and child dfs')
        # Step 4: Build parent rows
        parent_df = df[["instance","parent"] + parent_cols + graph_level_features].copy()
        parent_df = parent_df.rename(columns={"parent": "node"})
        parent_df.columns = [c.replace("parent_", "") for c in parent_df.columns]

        # Step 5: Build child rows
        child_df = df[["instance","child"] + child_cols + graph_level_features].copy()
        child_df = child_df.rename(columns={"child": "node"})
        child_df.columns = [c.replace("child_", "") for c in child_df.columns]
        print('combining and deduplicating')
        # Step 6: Combine both and aggregate duplicates
        node_df = pd.concat([parent_df, child_df], ignore_index=True)
        
        # --- Check for duplicates with differing values ---
        if debug_time:
            dupes = node_df[node_df.duplicated(["instance","node"], keep=False)].sort_values("node")
            if not dupes.empty:
                diff_nodes = []
                for n, g in dupes.groupby(["instance","node"]):
                    if not g.drop(columns=["instance","node"]).nunique().eq(1).all():
                        diff_nodes.append(n)
                        print("dupe detected: ", g)
                if diff_nodes:
                    print(f"⚠️ Warning: nodes with inconsistent duplicate values: {diff_nodes}")

        
        nan_rows = node_df[node_df.isna().any(axis=1)]
        if debug_time:
            if not nan_rows.empty:
                print("⚠️ Warning: rows with NaN values detected:")
                print(nan_rows)
        node_df = node_df.groupby(["instance","node"]).mean().reset_index().fillna(0)
        node_df = node_df.sort_values(["instance","node"]).reset_index(drop=True)
        return node_df

def get_edge_features(res_feat_df, edge_level_features, instance, edge_labels=[]):
    instance_name = str(instance['name']).split('/')[-1].split('.')[0]
    
    # Fix: assign sorted result back to edges
    edges = [(int(a), int(b)) for (a, b) in instance['precedence_relations']]
    edges = sorted(edges) 
    torch_edges = [ (a-1, b-1) for (a,b) in edges] #0 based indexes
    edge_index = torch.tensor(torch_edges, dtype=torch.long).t().contiguous()
    
    inst_df = res_feat_df[res_feat_df['instance'] == instance_name].copy()
    inst_df = inst_df[edge_level_features + edge_labels + ["edge"]]
    inst_df['edge_tuple'] = inst_df['edge'].apply(lambda x: tuple(map(int, ast.literal_eval(x))))
    
    leftover = set(inst_df['edge_tuple']) - set(edges)
    if leftover:
        print(f"WARNING: {len(leftover)} edges not in sorted list: {leftover}")
    
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    df_sorted = inst_df[inst_df['edge_tuple'].isin(edges)].copy()
    df_sorted = df_sorted.iloc[df_sorted['edge_tuple'].map(edge_to_idx).argsort()]
    df_sorted = df_sorted.drop(columns=['edge_tuple', 'edge'])  # Drop both helper columns
    
    if len(edge_labels) > 0:
        edge_label_tensor = torch.tensor(df_sorted[edge_labels].values, dtype=torch.float)  # Add .values
    else:
        edge_label_tensor = None
    
    edge_features = torch.tensor(df_sorted[edge_level_features].values, dtype=torch.float)  # Add .values
    
    return edge_index, edge_features, edge_label_tensor
        
def merge_node_data(node_feat_df, salbp_inst, debugging=False):
    '''This takes the existing instance results and gets their data for an instance'''
  #Treat data for NN use
    # If nodes appear multiple times, aggregate (mean or other)
    instance_name = str(salbp_inst['name']).split('/')[-1].split('.')[0]
    print('doing', instance_name)
    node_df = node_feat_df[node_feat_df['instance']== instance_name].copy()
    node_df.drop(columns=['instance'], inplace=True)
        
    task_df = pd.DataFrame(list(salbp_inst['task_times'].items()), columns=["node", "value"])
    task_df["node"] = task_df["node"].astype(int)
    node_df["node"] = node_df["node"].astype(int)
    task_df = task_df.merge(node_df, on="node", how= "left").fillna(0)
    if debugging:
        nan_rows = task_df[task_df.isna().any(axis=1)]
        if not nan_rows.empty:
            print("⚠️ Warning: rows with NaN values detected for the joined dataframe:")
            print(nan_rows)
            return
    
    # Sort rows by node ID
    task_df = task_df.sort_values("node").reset_index(drop=True)
    assert task_df["node"].is_monotonic_increasing, "Nodes not sorted properly!"
    assert task_df["node"].nunique() == len(salbp_inst["task_times"].keys()), "Node mismatch!"
    assert task_df["node"].min() == 0 or task_df["node"].min() == 1, "Unexpected node numbering!"
    # Extract feature matrix (drop the node ID column)
    x_cols = list(task_df.drop(columns=["node"]).columns)
    x_values = task_df.drop(columns=["node"]).values

    # Convert to a PyTorch tensor (float type)
    x = torch.tensor(x_values, dtype=torch.float)
    return x,x_cols, instance_name

def get_edge_and_node_data_nn(
    alb, graph_data, G_max_close=None, n_random_solves=0,
    feature_types={"all"}
):
    """Gets edge and graph data for an instance for use with neural net"""

    total_start = time.time()

    # --- Timing container ---
    timings = {}

    # ----------------- Graph setup -----------------
    t0 = time.time()
    edge_list = []
    G = nx.DiGraph()
    G.add_nodes_from([str(i) for i in range(1, alb['num_tasks'] + 1)])
    G.add_edges_from(alb['precedence_relations'])
    nx.set_node_attributes(G, {i: {'task_time': alb['task_times'][str(i)]} for i in G.nodes})

    if G_max_close:
        nx.set_node_attributes(G_max_close, {i: {'task_time': alb['task_times'][str(i)]} 
                                             for i in G_max_close.nodes})
    timings['graph_setup'] = time.time() - t0

    # ----------------- Positional weights -----------------
    t0 = time.time()
    positional_weights = get_all_positional_weight(G, G_max_close=G_max_close)
    timings['positional_weights'] = time.time() - t0

    # ----------------- Longest chains -----------------
    t0 = time.time()
    longest_chains_to, longest_chains_from = longest_weighted_chains(G)
    chain_stats = precompute_chain_stats(longest_chains_to, longest_chains_from, 
                                     alb['precedence_relations'])
    timings['longest_chains'] = time.time() - t0
    
    # ----------------- Edge neighbor stats -----------------
    t0 = time.time()
    edge_weights = get_edge_neighbor_max_min_avg_std(G)
    timings['edge_weight_stats'] = time.time() - t0

    # ----------------- Random walk stats -----------------
    if not feature_types.isdisjoint({'all', 'rw'}):
        t0 = time.time()
        walk_info = generate_walk_stats_ALB(G, num_walks=5, walk_length=10)
        timings['walk_stats'] = time.time() - t0

    # ----------------- GraphEval: load stats -----------------
    if not feature_types.isdisjoint({'all', 'grapheval'}):
        t0 = time.time()
        load_stats = graph_data.pop('load_stats')
        timings['load_stats_extract'] = time.time() - t0

    # -------------------------------------------------------
    #                  Per-node processing
    # -------------------------------------------------------
    timings['t_neighborhood'] = 0
    timings['chain'] = 0
    timings['degree'] = 0
    node_dict = {}
    for node,value in alb['task_times'].items():
        current_node = {}
        current_node['in_degree']= G.in_degree(node)
        current_node['out_degree']=G.out_degree(node)
        current_node['stage'] = len(longest_chains_to[node]['nodes'])
        current_node['pos_weight'] = positional_weights[node]
        current_node['weight'] = value
        current_node['value'] = value #Backwards compatibility

        
                # ----------- GraphEval load data ----------
        if not feature_types.isdisjoint({'all', 'grapheval'}):
            t0 = time.time()
            parent_load_data = get_load_data(load_stats, node, 'load_' )
            
            current_node.update(parent_load_data)
        # ----------- Random walk features ----------
        if not feature_types.isdisjoint({'all', 'rw'}):
            t0 = time.time()
            p = walk_info[walk_info['node'] == node].drop(columns=['node']).to_dict('records')[0]
            current_node.update(p)
        node_dict[node] = current_node
    # -------------------------------------------------------
    #                  Per-edge processing
    # -------------------------------------------------------

    for idx, edge in enumerate(alb['precedence_relations']):
        parent_data = {} #Note that some of this data will be redundant
        child_data = {}
        edge_start = time.time()

        # Neighborhood features
        t0 = time.time()
        neighborhood_data = get_neighborhood_edge_stats(edge_weights[tuple(edge)])
        t_neighborhood = time.time() - t0
        timings['t_neighborhood'] += t_neighborhood
        # Longest-chain features
        t0 = time.time()
        stats = chain_stats[tuple(edge)]
        chain_avg = stats['avg']
        chain_min = stats['min']
        chain_max = stats['max']
        chain_std = stats['std']
        stage_diff = abs(len(longest_chains_to[edge[0]]['nodes'])-len(longest_chains_to[edge[1]]['nodes']))
        t_chain = time.time() - t0
        timings['chain'] += t_chain
        # Degree + stage + positional weight
        t0 = time.time()
        t_degrees = time.time() - t0
        timings['degree'] += t_degrees
        res_dict = {
             **neighborhood_data,
            'edge': edge,
            'idx': idx,
            'chain_avg': chain_avg,
            'chain_min': chain_min,
            'chain_max': chain_max,
            'chain_std': chain_std,
            'stage_difference': stage_diff,
            'edge_data_time': time.time() - edge_start
        }
        # ----------- EdgeEval solve ----------
        if not feature_types.isdisjoint({'all', 'edgeeval'}):
            t0 = time.time()
            n_stations = priority_edge_solves(edge, alb, n_random_solves)
            stations_delta = graph_data['priority_min_stations'] - n_stations
            res_dict['stations_delta'] = stations_delta
            res_dict['edge_solve_time'] = time.time() - t0

        edge_list.append(res_dict)

    #Print summary timings
    # print("\n=== Timing Summary ===")
    # for k, v in timings.items():
    #     print(f"{k:25s}: {v:8.4f} sec")

    # print(f"total_runtime           : {time.time() - total_start:8.4f} sec")
    # print("=======================\n")

    return edge_list, node_dict

def albp_to_features_nn(alb_instance, salbp_type="salbp_1", cap_constraint=None, G_max_red=None, G_max_close=None, n_random=100, n_edge_random=100, feature_types={"all"}, return_assignments = False):
    """This prepares the features for a neural network friendly format"""
    t_func_start = time.perf_counter()
    
    profile_stats = {
        'instance_parsing': 0.0,
        'get_graph_metrics': 0.0,
        'get_time_stats': 0.0,
        'generate_priority_sol_stats': 0.0,
        'dict_merging': 0.0,
        'get_combined_edge_and_graph_data': 0.0,
        'dataframe_conversion': 0.0,
        'function_total': 0.0
    }
    
    start = time.time()
    
    # Instance parsing
    t_start = time.perf_counter()
    instance = str(alb_instance['name']).split('/')[-1].split('.')[0]
    profile_stats['instance_parsing'] = time.perf_counter() - t_start
    
    # Get graph metrics
    t_start = time.perf_counter()
    graph_metrics = get_graph_metrics(alb_instance,G_max_close, G_max_red)
    profile_stats['get_graph_metrics'] = time.perf_counter() - t_start
    
    if salbp_type == "salbp_1":
        graph_data = {'instance': instance}
        
        # Get time stats
        t_start = time.perf_counter()
        time_metrics = get_time_stats(alb_instance, C=cap_constraint)
        profile_stats['get_time_stats'] = time.perf_counter() - t_start
        
        # Generate priority solution stats (conditional)
        if not feature_types.isdisjoint({'all', 'grapheval'}):
            t_start = time.perf_counter()
            combined_metrics = generate_priority_sol_stats_salbp1(alb_instance, n_random=n_random, return_assignments=return_assignments )
            profile_stats['generate_priority_sol_stats'] = time.perf_counter() - t_start
            
            # Dict merging
            t_start = time.perf_counter()
            graph_data = {**graph_data, **combined_metrics}
            profile_stats['dict_merging'] += time.perf_counter() - t_start
        
        graph_time = time.time() - start
        
        # Final dict merging
        t_start = time.perf_counter()
        graph_data = {**graph_data, **time_metrics, **graph_metrics, 'global_feature_time': graph_time}
        profile_stats['dict_merging'] += time.perf_counter() - t_start
        
        # Get combined edge and graph data
        t_start = time.perf_counter()
        edge_data, node_data = get_edge_and_node_data_nn(alb_instance, graph_data, G_max_close=G_max_close, feature_types=feature_types, n_random_solves=n_edge_random)
        profile_stats['get_combined_edge_and_graph_data'] = time.perf_counter() - t_start
              
    elif salbp_type == "salbp_2":  # TODO check if this works
        if cap_constraint:
            alb_instance['n_stations'] = cap_constraint
        
        # Get time stats for SALBP-2
        t_start = time.perf_counter()
        time_metrics = get_time_stats_salb2(alb_instance, S=cap_constraint)
        profile_stats['get_time_stats'] = time.perf_counter() - t_start
        
        # Generate priority solution stats for SALBP-2
        t_start = time.perf_counter()
        combined_metrics = generate_priority_sol_stats_salbp2(alb_instance)
        profile_stats['generate_priority_sol_stats'] = time.perf_counter() - t_start
        
        # Dict merging
        t_start = time.perf_counter()
        graph_data = {'instance': instance, **time_metrics, **graph_metrics, **combined_metrics}
        profile_stats['dict_merging'] += time.perf_counter() - t_start
        
        # Get combined edge and graph data
        t_start = time.perf_counter()
        edge_data, node_data = get_combined_edge_and_graph_data(alb_instance, graph_data)
        profile_stats['get_combined_edge_and_graph_data'] = time.perf_counter() - t_start
    
    # Convert to DataFrame
    
    # Print profile summary
    # print("\n=== PROFILE: albp_to_features ===")
    # print(f"Total time: {profile_stats['function_total']:.4f}s")
    # print(f"SALBP type: {salbp_type}")
    # print(f"  instance_parsing:                   {profile_stats['instance_parsing']:.4f}s ({profile_stats['instance_parsing']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  get_graph_metrics:                  {profile_stats['get_graph_metrics']:.4f}s ({profile_stats['get_graph_metrics']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  get_time_stats:                     {profile_stats['get_time_stats']:.4f}s ({profile_stats['get_time_stats']/profile_stats['function_total']*100:.1f}%)")
    # if profile_stats['generate_priority_sol_stats'] > 0:
    #     print(f"  generate_priority_sol_stats:        {profile_stats['generate_priority_sol_stats']:.4f}s ({profile_stats['generate_priority_sol_stats']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  dict_merging:                       {profile_stats['dict_merging']:.4f}s ({profile_stats['dict_merging']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  get_combined_edge_and_graph_data:   {profile_stats['get_combined_edge_and_graph_data']:.4f}s ({profile_stats['get_combined_edge_and_graph_data']/profile_stats['function_total']*100:.1f}%)")
    # print(f"  dataframe_conversion:               {profile_stats['dataframe_conversion']:.4f}s ({profile_stats['dataframe_conversion']/profile_stats['function_total']*100:.1f}%)")
    # print(f"Rows in result: {len(final_data)}, n_random solutions: {n_random}")
    # print("=" * 35)
    
    return graph_data, node_data, edge_data


def sort_dict_by_list(d, order_list, warning=True):
    """Sort dictionary by order specified in list, warn if keys missing."""
    sorted_dict = {}
    for key in order_list:
        if key in d:
            sorted_dict[key] = d[key]
        elif warning:

            print(f"Warning: Key '{key}' not found in dictionary")
    return sorted_dict


def get_x_tensor(node_data, graph_data,x_features):
    data_list = []
    #First sorts nodes by number
    node_data = dict(sorted(node_data.items(), key=lambda x: int(x[0])))
    for node, data in node_data.items():
            #sorts node and graph data by features vector
            n_and_g_data = sort_dict_by_list({**data, **graph_data}, x_features)
            n_and_g_data = list(n_and_g_data.values())
            data_list.append(n_and_g_data)
    return  torch.tensor(data_list, dtype=torch.float)


def edge_list_to_tensor(edge_data, edge_level_features, edge_label_df=None):
    edge_tensor = []
    edge_feature_tensor = []
    edge_label_names = None
    if edge_label_df is not None:
        edge_label_names = list(result.drop(columns=['edge']).columns)
        edge_label_df[['u','v']] = pd.DataFrame(edge_label_df['edge'].tolist(), index=edge_label_df.index)
        
    edge_labels = []
    
    for edge in edge_data:
        u,v = edge.pop('edge')
        if edge_label_df is not None:
            result = edge_label_df[(edge_label_df['u'] == int(u)) & (edge_label_df['v'] == int(v))].copy()
            values = result.drop(columns=['edge', 'u','v']).iloc[0].tolist()
            edge_labels.append(values)
        edge_features= sort_dict_by_list(edge, edge_level_features)
        edge_feature_tensor.append(list(edge_features.values()))
        edge_tensor.append([int(u)-1, int(v)-1])
    edge_feature_tensor = torch.tensor(edge_feature_tensor, dtype=torch.float)
    edge_tensor = torch.tensor(edge_tensor, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)
    return edge_tensor, edge_feature_tensor, edge_labels, edge_label_names




def geo_from_albp_dict(alb_instance, x_features, edge_level_features, y_graph=None, graph_label_cols=None, edge_label_df = None, salbp_type="salbp_1", cap_constraint=None, G_max_red=None, G_max_close=None, n_random=100, n_edge_random=100, feature_types={"all"}, return_assignments = False):
    instance_name = str(alb_instance['name']).split('/')[-1].split('.')[0]
    graph_data, node_data, edge_data = albp_to_features_nn(alb_instance, salbp_type=salbp_type, cap_constraint=cap_constraint, G_max_red=G_max_red, G_max_close=G_max_close, n_random=n_random, n_edge_random=n_edge_random, feature_types=feature_types, return_assignments = return_assignments)
    x= get_x_tensor(node_data,graph_data, x_features)
    edge_index, edge_features, y_edge, edge_labels = edge_list_to_tensor(edge_data, edge_level_features, edge_label_df)
    data = Data(
                instance_name = instance_name,
                x=x,
                x_cols = x_features,
                edge_index=edge_index,
                y=y_graph,
                graph_labels = graph_label_cols,
                y_edge = y_edge,
                edge_labels = edge_labels,
                edge_attr=edge_features,
                edge_cols=edge_level_features
            )
    # all_data = {'instance':instance_name, 'edges': edge_index, 'edge_features':edge_features,'edge_label_values':edge_label_values, 'edge_labels':edge_labels,'features':x, 
    #                           'graph_labels':graph_label_cols, 'graph_label_values':graph_labels, 'x_cols':x_features,'node_level_features':None,
    #                           'graph_level_features':None, 'edge_level_features':edge_level_features}
    return data


def generate_geo_ready(instance_pkl_fp, res_feat_fp, node_level_features, graph_level_features, edge_level_features, graph_label_cols=['orig_n_stations'], edge_labels=[]):
    alb_dicts = open_salbp_pickle(instance_pkl_fp)[:1]
    res_feat_df = pd.read_csv(res_feat_fp)
    node_feat_df = get_node_and_graph_feat(res_feat_df, node_level_features, graph_level_features, debug_time = False)
    instance_data = []
    for instance in alb_dicts:

        x,x_cols, instance_name =  merge_node_data(node_feat_df, instance, debugging=False)
        edge_index, edge_features, edge_label_values = get_edge_features(res_feat_df,edge_level_features, instance, edge_labels=edge_labels)
        obj_vals = res_feat_df[res_feat_df['instance'] == instance_name][graph_label_cols].iloc[0]
        graph_labels = torch.tensor(obj_vals.values, dtype=torch.float)
        data_dict = {'instance':instance_name, 'edges': edge_index, 'edge_features':edge_features,'edge_label_values':edge_label_values, 'edge_labels':edge_labels,'features':x, 
                              'graph_labels':graph_label_cols, 'graph_label_values':graph_labels, 'x_cols':x_cols,'node_level_features':node_level_features,
                              'graph_level_features':graph_level_features, 'edge_level_features':edge_level_features}
        instance_data.append(data_dict)
    return instance_data


def process_one(args):
    n, ds,node_level_feats,graph_level_feats,edge_level_feats = args

    pereria_dataset_fp = f"/home/jot240/DADA/DADA/data/results/{ds}_{n}/{ds}_{n}_orig_results_edge.csv"
    instance_pkl_fp    = f"/home/jot240/DADA/DADA/data/raw/pkl_datasets/n_{n}_{ds}.pkl"

    geo_ready = generate_geo_ready(
        instance_pkl_fp,
        pereria_dataset_fp,
        node_level_feats,
        graph_level_feats,
        edge_level_feats
    )

    out_fp = f"/home/jot240/DADA/DADA/data/pereria_results/pytorch_ready/{ds}_n_{n}_geo_ready.pkl"
    with open(out_fp, "wb") as f:
        pickle.dump(geo_ready, f)

    return f"done {ds}_{n}"

def main():
    neural_network_node_features =  [
                        'pos_weight',
                        'stage',
                        'in_degree',
                        'out_degree',
                        'load_mean',
                        'load_max',
                        'load_min',
                        'load_std',
                        'rw_mean_total_time',
                        'rw_mean_min_time',
                        'rw_mean_max_time',
                        'rw_mean_n_unique_nodes',
                        'rw_mean_walk_length',
                        'rw_min',
                        'rw_max',
                        'rw_mean',
                        'rw_std',
                        'rw_n_unique_nodes',
]
    all_features =  ['instance',
                        'priority_min_stations',
                        'priority_max_stations',
                        'priority_min_gap',
                        'priority_max_gap',
                        'random_spread',
                        'random_coefficient_of_variation',
                        'random_avg_gap',
                        'random_min_gap',
                        'random_max_gap',
                        'random_avg_efficiency',
                        'priority_calc_time',
                        'min_div_c',
                        'max_div_c',
                        'sum_div_c',
                        'std_div_c',
                        't_cv',
                        'ti_size',
                        'avg_div_c',
                        'lb_6',
                        'n_edges',
                        'order_strength',
                        'average_number_of_immediate_predecessors',
                        'max_degree',
                        'max_in_degree',
                        'max_out_degree',
                        'divergence_degree',
                        'convergence_degree',
                        'n_bottlenecks',
                        'share_of_bottlenecks',
                        'avg_degree_of_bottlenecks',
                        'n_chains',
                        'avg_chain_length',
                        'nodes_in_chains',
                        'n_stages',
                        'stages_div_n',
                        'prec_strength',
                        'prec_bias',
                        'prec_index',
                        'n_isolated_nodes',
                        'share_of_isolated_nodes',
                        'n_tasks_without_predecessors',
                        'share_of_tasks_without_predecessors',
                        'avg_tasks_per_stage',
                        'graph_feature_time',
                        'global_feature_time',
                        'neighborhood_min',
                        'neighborhood_max',
                        'neighborhood_avg',
                        'neighborhood_std',
                        'child_min',
                        'child_max',
                        'child_avg',
                        'child_std',
                        'parent_min',
                        'parent_max',
                        'parent_avg',
                        'parent_std',
                        'parent_weight',
                        'child_weight',
                        'abs_weight_difference',
                        'edge',
                        'idx',
                        'parent_pos_weight',
                        'parent_stage',
                        'child_pos_weight',
                        'child_stage',
                        'stage_difference',
                        'parent_in_degree',
                        'parent_out_degree',
                        'child_in_degree',
                        'child_out_degree',
                        'chain_avg',
                        'chain_min',
                        'chain_max',
                        'chain_std',
                        'edge_data_time',
                        'load_parent_mean',
                        'load_parent_max',
                        'load_parent_min',
                        'load_parent_std',
                        'load_child_mean',
                        'load_child_max',
                        'load_child_min',
                        'load_child_std',
                        'load_data_time',
                        'rw_mean_total_time',
                        'rw_mean_min_time',
                        'rw_mean_max_time',
                        'rw_mean_n_unique_nodes',
                        'rw_mean_walk_length',
                        'rw_min',
                        'rw_max',
                        'rw_mean',
                        'rw_std',
                        'rw_n_unique_nodes',
                        'rw_elapsed_time',
                        'child_rw_mean_total_time',
                        'child_rw_mean_min_time',
                        'child_rw_mean_max_time',
                        'child_rw_mean_n_unique_nodes',
                        'child_rw_mean_walk_length',
                        'child_rw_min',
                        'child_rw_max',
                        'child_rw_mean',
                        'child_rw_std',
                        'child_rw_n_unique_nodes',
                        'child_rw_elapsed_time',
                        'walk_data_time',
                        'stations_delta',
                        'edge_solve_time',
                        'weight_sum',]

    graph_level_feats = ['priority_min_stations',
                        'priority_max_stations',
                        'priority_min_gap',
                        'priority_max_gap',
                        'random_spread',
                        'random_coefficient_of_variation',
                        'random_avg_gap',
                        'random_min_gap',
                        'random_max_gap',
                        'random_avg_efficiency',
                        'min_div_c',
                        'max_div_c',
                        'sum_div_c',
                        'std_div_c',
                        't_cv',
                        'ti_size',
                        'avg_div_c',
                        'lb_6',
                        'n_edges',
                        'order_strength',
                        'average_number_of_immediate_predecessors',
                        'max_degree',
                        'max_in_degree',
                        'max_out_degree',
                        'divergence_degree',
                        'convergence_degree',
                        'n_bottlenecks',
                        'share_of_bottlenecks',
                        'avg_degree_of_bottlenecks',
                        'n_chains',
                        'avg_chain_length',
                        'nodes_in_chains',
                        'n_stages',
                        'stages_div_n',
                        'prec_strength',
                        'prec_bias',
                        'prec_index',
                        'n_isolated_nodes',
                        'share_of_isolated_nodes',
                        'n_tasks_without_predecessors',
                        'share_of_tasks_without_predecessors',
                        'avg_tasks_per_stage']

    node_level_feats =  [
                        'parent_pos_weight',
                        'parent_stage',
                        'child_pos_weight',
                        'child_stage',
                        'parent_in_degree',
                        'parent_out_degree',
                        'child_in_degree',
                        'child_out_degree',
                        'load_parent_mean',
                        'load_parent_max',
                        'load_parent_min',
                        'load_parent_std',
                        'load_child_mean',
                        'load_child_max',
                        'load_child_min',
                        'load_child_std',
                        'rw_mean_total_time',
                        'rw_mean_min_time',
                        'rw_mean_max_time',
                        'rw_mean_n_unique_nodes',
                        'rw_mean_walk_length',
                        'rw_min',
                        'rw_max',
                        'rw_mean',
                        'rw_std',
                        'rw_n_unique_nodes',
                        'child_rw_mean_total_time',
                        'child_rw_mean_min_time',
                        'child_rw_mean_max_time',
                        'child_rw_mean_n_unique_nodes',
                        'child_rw_mean_walk_length',
                        'child_rw_min',
                        'child_rw_max',
                        'child_rw_mean',
                        'child_rw_std',
                        'child_rw_n_unique_nodes',]

    edge_level_feats= [
                            'child_min',
                        'child_max',
                        'child_avg',
                        'child_std',
                        'parent_min',
                        'parent_max',
                        'parent_avg',
                        'parent_std',
                        'neighborhood_min',
                        'neighborhood_max',
                        'neighborhood_avg',
                        'neighborhood_std',
                        'abs_weight_difference',
                        'stage_difference',
                        'chain_avg',
                        'chain_min',
                        'chain_max',
                        'chain_std',
                        'stations_delta',
                        'weight_sum',]

    n_range = [70, 80, 125, 150]
    datasets = ["unstructured", "chains", "bottleneck"]
    #n_range = [50]
    datasets = ["unstructured"]

    args = [( n, ds,node_level_feats,graph_level_feats,edge_level_feats ) for n in n_range for ds in datasets]

    with Pool() as pool:
        for msg in pool.imap_unordered(process_one, args):
            print(msg)
if __name__ == "__main__":
    main()

