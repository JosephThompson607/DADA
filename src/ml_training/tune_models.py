import glob
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import xgboost as xgb

src_path = os.path.abspath("src")
if src_path not in sys.path:
    sys.path.append(src_path)
from alb_instance_compressor import *
from SALBP_solve import *
from copy import deepcopy
from metrics.node_and_edge_features import *
from metrics.graph_features import *
from metrics.time_metrics import *
#from data_prep import *
import multiprocessing
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


from ml_models import *


def data_preprocessing_salbp1(df, s_orig_col = 's_orig' ):
    # min_and_max = df.groupby('instance').agg({'no_stations': ['min']}).reset_index()
    # min_and_max.columns = ['instance', 'min',]
    # df = pd.merge(df, min_and_max, on='instance', how='left')
    # print(df.columns)
    df["n_edges"] = df["average_number_of_immediate_predecessors"] * df['n_edges']
    df["n_edges"] = df["n_edges"].astype(int)
    df["avg_div_c"] = df['sum_div_c']/df["n_tasks"]
    df["stages_div_n"] = df['n_stages']/df["n_tasks"]
    df['min_less_max'] = df['min'] < df[s_orig_col]
    df['bin_less_max'] = df['bin_lb'] < df[s_orig_col]
    df['is_less_max'] = df['no_stations'] < df[s_orig_col]
    df['min_less_max'] = df['min_less_max'].astype(int)
    df['order_strength'] = df['order_strength'].astype(float)
    
    
    #counts the number of times the min is less than the max for each instance
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    only_1 = df.drop_duplicates(subset='instance', keep='first').copy()
    only_1= only_1.drop(columns = ['precedence_relation', 'no_stations','edge','parent_weight', 'parent_pos_weight',
       'child_weight', 'child_pos_weight', 'neighborhood_min',
       'neighborhood_max', 'neighborhood_avg', 'neighborhood_std',
       'parent_in_degree', 'parent_out_degree', 'child_in_degree',
       'child_out_degree', 'chain_avg', 'chain_min', 'chain_max', 'chain_std',
       'edge_data_time', 'rw_mean_total_time', 'rw_mean_min_time',
       'rw_mean_max_time', 'rw_mean_n_unique_nodes', 'rw_mean_walk_length',
       'rw_min', 'rw_max', 'rw_mean', 'rw_std', 'rw_n_unique_nodes',
       'rw_elapsed_time', 'child_rw_mean_total_time', 'child_rw_mean_min_time',
       'child_rw_mean_max_time', 'child_rw_mean_n_unique_nodes',
       'child_rw_mean_walk_length', 'child_rw_min', 'child_rw_max',
       'child_rw_mean', 'child_rw_std', 'child_rw_n_unique_nodes',
       'child_rw_elapsed_time', 'is_less_max'])
    only_1['OS_bin'] = pd.cut(only_1['order_strength'], bins)
    only_1['bin_diff'] = (only_1['bin_lb'].round() - np.ceil(only_1['sum_div_c']))/np.ceil(only_1['sum_div_c'])
    only_1['salbp_diff'] =  (only_1[s_orig_col] - np.ceil(only_1['sum_div_c']))/np.ceil(only_1['sum_div_c'])
    only_1['salb_bin_gap'] =(only_1[s_orig_col] -  only_1['bin_lb'])/np.ceil(only_1['sum_div_c'])

    return df, only_1

def load_miranda():
    bott_60 = pd.read_csv("data/results/bottleneck_60/bottleneck_60_ml_ready.csv")
    bott_60.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    bott_60['dataset'] = "bottleneck"
    bott_60['n_tasks'] = 60
    bott_60, bott_60_graph = data_preprocessing_salbp1(bott_60)

    bott_50 = pd.read_csv("data/results/bottleneck_50/bottleneck_50_ml_ready.csv")
    bott_50['dataset'] = "bottleneck"
    bott_50.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    bott_50['n_tasks'] = 50
    bott_50, bott_50_graph = data_preprocessing_salbp1(bott_50)

    chains_60 = pd.read_csv("data/results/chains_60/chains_60_ml_ready.csv")
    chains_60['dataset'] = "chains"
    chains_60.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    chains_60['n_tasks'] = 60
    chains_60, chains_60_graph = data_preprocessing_salbp1(chains_60)

    chains_50 = pd.read_csv("data/results/chains_50/chains_50_ml_ready.csv")
    chains_50['dataset'] = "chains"
    chains_50.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    chains_50['n_tasks'] = 50
    chains_50, chains_50_graph = data_preprocessing_salbp1(chains_50)

    unstructured_60 = pd.read_csv("data/results/unstructured_60/unstructured_60_ml_ready.csv")
    unstructured_60['dataset'] = "unstructured"
    unstructured_60.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    unstructured_60['n_tasks'] = 60
    unstructured_60, unstructured_60_graph = data_preprocessing_salbp1(unstructured_60)

    unstructured_50 = pd.read_csv("data/results/unstructured_50/unstructured_50_ml_ready.csv")
    unstructured_50['dataset'] = "unstructured"
    unstructured_50.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    unstructured_50['n_tasks'] = 50
    unstructured_50, unstructured_50_graph = data_preprocessing_salbp1(unstructured_50)

    print("moving on to larger instances")
    bott_100 = pd.read_csv("data/results/bottleneck_100/bottleneck_100_ml_ready.csv")
    bott_100['dataset'] = "bottleneck"
    bott_100.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    bott_100['n_tasks'] = 100
    bott_100, bott_100_graph = data_preprocessing_salbp1(bott_100)


    bott_90 = pd.read_csv("data/results/bottleneck_90/bottleneck_90_ml_ready.csv")
    bott_90['dataset'] = "bottleneck"
    bott_90.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    bott_90['n_tasks'] = 90
    bott_90, bott_90_graph = data_preprocessing_salbp1(bott_90)

    chains_100 = pd.read_csv("data/results/chains_100/chains_100_ml_ready.csv")
    chains_100['dataset'] = "chains"
    chains_100.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    chains_100['n_tasks'] = 100
    chains_100, chains_100_graph = data_preprocessing_salbp1(chains_100)


    chains_90 = pd.read_csv("data/results/chains_90/chains_90_ml_ready.csv")
    chains_90['dataset'] = "chains"
    chains_90.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    chains_90['n_tasks'] = 90
    chains_90, chains_90_graph = data_preprocessing_salbp1(chains_90)


    unstructured_100 = pd.read_csv("data/results/unstructured_100/unstructured_100_ml_ready.csv")
    unstructured_100['dataset'] = "unstructured"
    unstructured_100.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    unstructured_100['n_tasks'] = 100
    unstructured_100, unstructured_100_graph = data_preprocessing_salbp1(unstructured_100)

    unstructured_90 = pd.read_csv("data/results/unstructured_90/unstructured_90_ml_ready.csv")
    unstructured_90['dataset'] = "unstructured"
    unstructured_90.rename(columns = {'orig_stations':'s_orig'}, inplace=True)
    unstructured_90['n_tasks'] = 90
    unstructured_90, unstructured_90_graph = data_preprocessing_salbp1(unstructured_90)
    # print("combining dataframes")
    df_list = [bott_100, bott_90, chains_100, chains_90, unstructured_100, unstructured_90, bott_60, bott_50, chains_60, chains_50, unstructured_60, unstructured_50]
    #df_list = [bott_100, bott_90, chains_100, chains_90, unstructured_100, unstructured_90, bott_60,chains_60, unstructured_60]

    #df_list = [bott_100, bott_90, chains_100, chains_90, unstructured_100, unstructured_90]
    df_list_graph = [bott_100_graph, bott_90_graph, chains_100_graph, chains_90_graph, 
               unstructured_100_graph, unstructured_90_graph, bott_60_graph, bott_50_graph, 
               chains_60_graph, chains_50_graph, unstructured_60_graph, unstructured_50_graph]
    #only large instances
    # df_list_graph = [bott_100_graph, bott_90_graph, chains_100_graph, chains_90_graph, 
    #            unstructured_100_graph, unstructured_90_graph]
    # df_list_graph = [bott_100_graph, bott_90_graph, chains_100_graph, chains_90_graph, 
    #            unstructured_100_graph, unstructured_90_graph, bott_60_graph,  
    #            chains_60_graph,  unstructured_60_graph]
    combined_graph = pd.concat(df_list_graph)
    combined = pd.concat(df_list)
    return combined_graph, combined


def main():
    continuous_features_edge = [
                            'parent_weight', 'parent_pos_weight',
                                'child_weight', 'child_pos_weight', 'neighborhood_min',
                                'neighborhood_max', 'neighborhood_avg', 'neighborhood_std',
                                'parent_in_degree', 'parent_out_degree', 'child_in_degree',
                                'child_out_degree', 'chain_avg', 'chain_min', 'chain_max', 'chain_std',
                                'edge_data_time', 'rw_mean_total_time', 'rw_mean_min_time',
                                'rw_mean_max_time', 'rw_mean_n_unique_nodes',
                                'rw_min', 'rw_max', 'rw_mean', 'rw_std', 'rw_n_unique_nodes', 'child_rw_mean_total_time', 'child_rw_mean_min_time',
                                'child_rw_mean_max_time', 'child_rw_mean_n_unique_nodes', 'child_rw_min', 'child_rw_max',
                                'child_rw_mean', 'child_rw_std', 'child_rw_n_unique_nodes', 'priority_min_gap', 'priority_max_gap',
                                'random_spread', 'random_coefficient_of_variation', 'random_avg_gap',
                                'random_min_gap', 'random_max_gap', 
                                'min_div_c', 'max_div_c',  'std_div_c','avg_div_c',
                                'order_strength', 'average_number_of_immediate_predecessors',
                                'max_degree', 'max_in_degree', 'max_out_degree', 'divergence_degree',
                                'convergence_degree',  'share_of_bottlenecks',
                                    'avg_chain_length',
                                'nodes_in_chains', 'stages_div_n', 'n_isolated_nodes',
                                'share_of_isolated_nodes', 'n_tasks_without_predecessors',
                                'share_of_tasks_without_predecessors', 'avg_tasks_per_stage',
                            ] 


    #removed_graph = ["'n_chains','n_bottlenecks','avg_degree_of_bottlenecks', 
    continuous_features_graph = [
        'min_div_c', 'max_div_c',  'std_div_c','avg_div_c',
        'order_strength', 'average_number_of_immediate_predecessors',
        'max_degree', 'max_in_degree', 'max_out_degree', 'divergence_degree', 'priority_min_gap', 'priority_max_gap',
        'random_spread', 'random_coefficient_of_variation', 'random_avg_gap',
        'random_min_gap', 'random_max_gap', 
        'convergence_degree',  'share_of_bottlenecks',
            'avg_chain_length',
        'nodes_in_chains', 'stages_div_n', 'n_isolated_nodes',
        'share_of_isolated_nodes', 'n_tasks_without_predecessors',
        'share_of_tasks_without_predecessors', 'avg_tasks_per_stage',

    ]
    print("loading data")
    combined_graph, combined = load_miranda()
    df = combined
    X = df[continuous_features_edge]
    y = df['is_less_max']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_2 = combined_graph[continuous_features_graph]
    y_2 = combined_graph['min_less_max']  # Target variable

    
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42, stratify=y_2)
    # print("training random forest")
    # rf_tuned_edge, rf_edge_params = random_forest_grid_search(X_train, y_train, X_test, y_test, res_name="rf_edge.json")
    # print("best rf edge params ", rf_edge_params )
    # print("trainging random forest graph")
    # rf_tuned_graph, rf_graph_params = random_forest_grid_search(    X_train_2,y_train_2, X_test_2,  y_test_2, res_name = "rf_graph.json" )
    # print("best rf graph params ", rf_graph_params)
    print("training xg boost")
    xg_tuned_edge, xg_edge_params = xgboost_grid_search(X_train, y_train, X_test, y_test, res_name="xg_edge.json")
    print("best xg boost edge params ", xg_edge_params)
    print("training xg boost graph" )
    xg_tuned_graph, xg_graph_params = xgboost_grid_search(X_train_2, y_train_2, X_test_2, y_test_2, res_name="xg_graph.json")
    print("best xg boost graph params",xg_graph_params )
    dt_tuned_edge, dt_edge_params = decision_tree_grid_search(X_train, y_train, X_test, y_test, res_name="dt_edge.json")
    print("best decision tree edge params ", dt_edge_params)
    print("training decision tree graph" )
    dt_tuned_graph, dt_graph_params = decision_tree_grid_search(X_train_2, y_train_2, X_test_2, y_test_2, res_name="dt_graph.json")
    print("best decision tree graph params", dt_graph_params)
    return




if __name__ == "__main__":
    main()
