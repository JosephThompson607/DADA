import glob
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import ast

# Assuming 'src' is in the same directory as your notebook
src_path = os.path.abspath("../../src")
if src_path not in sys.path:
    sys.path.append(src_path)
src_path = os.path.abspath("../../src/torch")
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
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
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
    x_cols = task_df.drop(columns=["node"]).columns
    x_values = task_df.drop(columns=["node"]).values

    # Convert to a PyTorch tensor (float type)
    x = torch.tensor(x_values, dtype=torch.float)
    return x,x_cols, instance_name


def generate_geo_ready(instance_pkl_fp, res_feat_fp, node_level_features, graph_level_features, edge_level_features, graph_label_cols=['orig_n_stations'], edge_labels=[]):
    alb_dicts = open_salbp_pickle(instance_pkl_fp)[:100]
    res_feat_df = pd.read_csv(res_feat_fp)
    node_feat_df = get_node_and_graph_feat(res_feat_df, node_level_features, graph_level_features, debug_time = False)
    instance_data = []
    for instance in alb_dicts:
        x,x_cols, instance_name =  merge_node_data(node_feat_df, instance, debugging=False)
        edge_index, edge_features, edge_label_values = get_edge_features(res_feat_df,edge_level_features, instance, edge_labels=edge_labels)
        obj_vals = res_feat_df[res_feat_df['instance'] == instance_name][graph_label_cols].iloc[0]
        graph_labels = torch.tensor(obj_vals.values, dtype=torch.float)
        instance_data.append({'instance':instance_name, 'edges': edge_index, 'edge_features':edge_features,'edge_label_values':edge_label_values, 'edge_labels':edge_labels,'features':x, 
                              'graph_labels':graph_labels, 'x_cols':x_cols,'node_level_features':node_level_features,
                              'graph_level_features':graph_level_features, 'edge_level_features':edge_level_features})
    return instance_data
def main():
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

    n = 50
    ds = "unstructured"
    pereria_dataset_fp = f"/home/jot240/DADA/DADA/data/results/{ds}_{n}/{ds}_{n}_orig_results_edge.csv"
    instance_pkl_fp = f"/home/jot240/DADA/DADA/data/raw/pkl_datasets/n_{n}_{ds}.pkl"
    geo_ready = generate_geo_ready(instance_pkl_fp, pereria_dataset_fp, node_level_feats, graph_level_feats, edge_level_feats)
    with open(f'/home/jot240/DADA/DADA/data/pereria_results/pytorch_ready/{ds}_n_{n}_geo_ready.pkl', 'wb') as f:
        pickle.dump(geo_ready, f)
    print("done")
if __name__ == "__main__":
    main()

