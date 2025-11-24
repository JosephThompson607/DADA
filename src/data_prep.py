import glob
import os
import sys
src_path = os.path.abspath("src")
if src_path not in sys.path:
    sys.path.append(src_path)

import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from alb_instance_compressor import *
from copy import deepcopy
from metrics.node_and_edge_features import *
from metrics.graph_features import *
from metrics.time_metrics import *
from metrics.global_combined_features import *
import multiprocessing



#FEATURE GENERATION FROM GRAPH
def alb_to_graph_data(alb_instance, salbp_type="salbp_1", cap_constraint = None):
    instance = str(alb_instance['name']).split('/')[-1].split('.')[0]
    print("processing instance", instance)
    if salbp_type == "salbp_1":
        time_metrics = get_time_stats(alb_instance, C=cap_constraint)
        combined_metrics = generate_priority_sol_stats_salbp1(alb_instance, generate_task_load_stats=False)
    elif salbp_type == "salbp_2":
        if cap_constraint:
            alb_instance['n_stations'] = cap_constraint
        time_metrics = get_time_stats_salb2(alb_instance, S=cap_constraint)
        combined_metrics = generate_priority_sol_stats_salbp2(alb_instance, generate_task_load_stats=False)
    graph_metrics = get_graph_metrics(alb_instance)
    graph_data = {'instance':instance, **time_metrics, **graph_metrics, **combined_metrics}
    return graph_data


def albp_to_features(alb_instance, salbp_type="salbp_1", cap_constraint=None, G_max_red=None, G_max_close=None, n_random=100, n_edge_random=100, feature_types={"all"}):
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
            combined_metrics = generate_priority_sol_stats_salbp1(alb_instance, n_random=n_random)
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
        final_data = get_combined_edge_and_graph_data(alb_instance, graph_data, G_max_close=G_max_close, feature_types=feature_types, n_random_solves=n_edge_random)
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
        final_data = get_combined_edge_and_graph_data(alb_instance, graph_data)
        profile_stats['get_combined_edge_and_graph_data'] = time.perf_counter() - t_start
    
    # Convert to DataFrame
    t_start = time.perf_counter()
    final_data = pd.DataFrame(final_data)
    profile_stats['dataframe_conversion'] = time.perf_counter() - t_start
    
    profile_stats['function_total'] = time.perf_counter() - t_func_start
    
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
    
    return final_data

# def albp_to_features(alb_instance, salbp_type="salbp_1", cap_constraint=None, n_random=100, n_edge_random=0, feature_types={"all"}):
#     t_func_start = time.perf_counter()
    
#     profile_stats = {
#         'instance_parsing': 0.0,
#         'get_graph_metrics': 0.0,
#         'get_time_stats': 0.0,
#         'generate_priority_sol_stats': 0.0,
#         'dict_merging': 0.0,
#         'get_combined_edge_and_graph_data': 0.0,
#         'dataframe_conversion': 0.0,
#         'function_total': 0.0
#     }
    
#     start = time.time()
    
#     # Instance parsing
#     t_start = time.perf_counter()
#     instance = str(alb_instance['name']).split('/')[-1].split('.')[0]
#     profile_stats['instance_parsing'] = time.perf_counter() - t_start
    
#     # Get graph metrics
#     t_start = time.perf_counter()
#     graph_metrics = get_graph_metrics(alb_instance)
#     profile_stats['get_graph_metrics'] = time.perf_counter() - t_start
    
#     if salbp_type == "salbp_1":
#         graph_data = {'instance': instance}
        
#         # Get time stats
#         t_start = time.perf_counter()
#         time_metrics = get_time_stats(alb_instance, C=cap_constraint)
#         profile_stats['get_time_stats'] = time.perf_counter() - t_start
        
#         # Generate priority solution stats (conditional)
#         if not feature_types.isdisjoint({'all', 'grapheval'}):
#             t_start = time.perf_counter()
#             combined_metrics = generate_priority_sol_stats_salbp1(alb_instance, n_random=n_random)
#             profile_stats['generate_priority_sol_stats'] = time.perf_counter() - t_start
            
#             # Dict merging
#             t_start = time.perf_counter()
#             graph_data = {**graph_data, **combined_metrics}
#             profile_stats['dict_merging'] += time.perf_counter() - t_start
        
#         graph_time = time.time() - start
        
#         # Final dict merging
#         t_start = time.perf_counter()
#         graph_data = {**graph_data, **time_metrics, **graph_metrics, 'global_feature_time': graph_time}
#         profile_stats['dict_merging'] += time.perf_counter() - t_start
        
#         # Get combined edge and graph data
#         t_start = time.perf_counter()
#         final_data = get_combined_edge_and_graph_data(alb_instance, graph_data, feature_types=feature_types, n_random_solves=n_edge_random)
#         profile_stats['get_combined_edge_and_graph_data'] = time.perf_counter() - t_start
        
#     elif salbp_type == "salbp_2":  # TODO check if this works
#         if cap_constraint:
#             alb_instance['n_stations'] = cap_constraint
        
#         # Get time stats for SALBP-2
#         t_start = time.perf_counter()
#         time_metrics = get_time_stats_salb2(alb_instance, S=cap_constraint)
#         profile_stats['get_time_stats'] = time.perf_counter() - t_start
        
#         # Generate priority solution stats for SALBP-2
#         t_start = time.perf_counter()
#         combined_metrics = generate_priority_sol_stats_salbp2(alb_instance)
#         profile_stats['generate_priority_sol_stats'] = time.perf_counter() - t_start
        
#         # Dict merging
#         t_start = time.perf_counter()
#         graph_data = {'instance': instance, **time_metrics, **graph_metrics, **combined_metrics}
#         profile_stats['dict_merging'] += time.perf_counter() - t_start
        
#         # Get combined edge and graph data
#         t_start = time.perf_counter()
#         final_data = get_combined_edge_and_graph_data(alb_instance, graph_data)
#         profile_stats['get_combined_edge_and_graph_data'] = time.perf_counter() - t_start
    
#     # Convert to DataFrame
#     t_start = time.perf_counter()
#     final_data = pd.DataFrame(final_data)
#     profile_stats['dataframe_conversion'] = time.perf_counter() - t_start
    
#     profile_stats['function_total'] = time.perf_counter() - t_func_start
    
#     # Print profile summary
#     print("\n=== PROFILE: albp_to_features ===")
#     print(f"Total time: {profile_stats['function_total']:.4f}s")
#     print(f"SALBP type: {salbp_type}")
#     print(f"  instance_parsing:                   {profile_stats['instance_parsing']:.4f}s ({profile_stats['instance_parsing']/profile_stats['function_total']*100:.1f}%)")
#     print(f"  get_graph_metrics:                  {profile_stats['get_graph_metrics']:.4f}s ({profile_stats['get_graph_metrics']/profile_stats['function_total']*100:.1f}%)")
#     print(f"  get_time_stats:                     {profile_stats['get_time_stats']:.4f}s ({profile_stats['get_time_stats']/profile_stats['function_total']*100:.1f}%)")
#     if profile_stats['generate_priority_sol_stats'] > 0:
#         print(f"  generate_priority_sol_stats:        {profile_stats['generate_priority_sol_stats']:.4f}s ({profile_stats['generate_priority_sol_stats']/profile_stats['function_total']*100:.1f}%)")
#     print(f"  dict_merging:                       {profile_stats['dict_merging']:.4f}s ({profile_stats['dict_merging']/profile_stats['function_total']*100:.1f}%)")
#     print(f"  get_combined_edge_and_graph_data:   {profile_stats['get_combined_edge_and_graph_data']:.4f}s ({profile_stats['get_combined_edge_and_graph_data']/profile_stats['function_total']*100:.1f}%)")
#     print(f"  dataframe_conversion:               {profile_stats['dataframe_conversion']:.4f}s ({profile_stats['dataframe_conversion']/profile_stats['function_total']*100:.1f}%)")
#     print(f"Rows in result: {len(final_data)}, n_random solutions: {n_random}")
#     print("=" * 35)
    
#     return final_data
# def albp_to_features(alb_instance, salbp_type="salbp_1", cap_constraint = None, n_random=100,n_edge_random=0, feature_types={"all"}):
#     start=time.time()
#     instance = str(alb_instance['name']).split('/')[-1].split('.')[0]
#     graph_metrics = get_graph_metrics(alb_instance)
#     if salbp_type == "salbp_1":
#         graph_data = {'instance':instance}
#         time_metrics = get_time_stats(alb_instance, C=cap_constraint)
#         if not feature_types.isdisjoint({'all', 'grapheval'}):
#             combined_metrics = generate_priority_sol_stats_salbp1(alb_instance, n_random=n_random)
#             graph_data = {**graph_data, **combined_metrics}
#         graph_time = time.time()-start
#         graph_data = {**graph_data, **time_metrics, **graph_metrics, 'global_feature_time': graph_time}
#         final_data = get_combined_edge_and_graph_data(alb_instance, graph_data, feature_types=feature_types, n_random_solves=n_edge_random)
#     elif salbp_type == "salbp_2": #TODO check if this works
#         if cap_constraint:
#             alb_instance['n_stations'] = cap_constraint
#         time_metrics = get_time_stats_salb2(alb_instance, S=cap_constraint)
#         combined_metrics = generate_priority_sol_stats_salbp2(alb_instance)
#         graph_data = {'instance':instance, **time_metrics, **graph_metrics, **combined_metrics}
#         final_data = get_combined_edge_and_graph_data(alb_instance, graph_data)

#     final_data = pd.DataFrame(final_data)
#     return final_data


def generate_graph_data_from_pickle(pickle_instance_fp, pool_size=4 , salbp_type="salbp_1",cap_constraint=None):
    alb_instances = open_salbp_pickle(pickle_instance_fp)
    with multiprocessing.Pool(pool_size) as pool:
        graph_data = pool.starmap(alb_to_graph_data, [(alb, salbp_type, cap_constraint) for alb in alb_instances])
    print(graph_data)
    graph_info_df = pd.DataFrame(graph_data)
    return graph_info_df



#FEATURE GENERATION FROM GRAPH
def alb_to_edge_data(alb_instance):
    instance_name = str(alb_instance['name']).split('/')[-1].split('.')[0]
    print("processing instance", instance_name)


    edge_data = get_edge_data(instance_name, alb_instance)
    edge_data = pd.DataFrame(edge_data)
    return edge_data



def generate_edge_data_from_pickle(pickle_instance_fp, pool_size=4 ):
    alb_instances = open_salbp_pickle(pickle_instance_fp)
    with multiprocessing.Pool(pool_size) as pool:
        edge_data = pool.map(alb_to_edge_data, alb_instances)
    edge_info_df = pd.concat(edge_data)

    return edge_info_df

def generate_all_features_from_pickle(pickle_instance_fp, pool_size=4 , salbp_type="salbp_1",cap_constraint=None):
    '''generates all features (edge, graph,). Must be called in order to use latest features'''
    alb_instances = open_salbp_pickle(pickle_instance_fp)
    with multiprocessing.Pool(pool_size) as pool:
        edge_data =  pool.starmap(albp_to_features, [(alb, salbp_type, cap_constraint) for alb in alb_instances])
    edge_info_df = pd.concat(edge_data)

    return edge_info_df



#PROCESSING SALBP OUTPUT
def process_file(filename, tolerance = 2):
    try:
        df = pd.read_csv(filename, index_col=None, header=0)
        salbp_upper_df = df.iloc[[0]]
        
     
        if not 'precedence_relation' in df.columns or not pd.isna(salbp_upper_df['precedence_relation'].iloc[0]):
            return None, {"instance": filename.name, "reason": "no_ub"}

        df = df.iloc[1:]
        df.columns = df.columns.str.strip()  # Just in case there are trailing spaces

       
        salbp_upper_df.rename(columns={"no_stations": "salbp_upper"}, inplace=True)
        salbp_upper_df = salbp_upper_df[["instance", "salbp_upper"]]

        df = pd.merge(df, salbp_upper_df, on="instance")

        if any(df["no_stations"] > df["salbp_upper"].astype(int)):
            return None, {"instance": filename.name, "reason": "upper bound mismatch"}
        if any(df["no_stations"] < df["bin_lb"].astype(int)):
            return None, {"instance": filename.name, "reason": "lower bound mismatch"}
        if len(df.index) + tolerance < df["original_n_precedence_constraints"].iloc[0]:
            return None, {"instance": filename.name, "reason": "missing_edge"}

        return df, None
    except Exception as e:
        return None, {"instance": filename.name, "reason": f"error: {e}"}

def process_SALBP_res_folder(path, max_workers=6):
    path = Path(path).expanduser()
    all_files = list(path.glob("*.csv"))
    good_data = []
    bad_instances = []
    

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file): file for file in all_files}
        for future in as_completed(futures):
            df, error = future.result()
            if df is not None:
                good_data.append(df)
            if error is not None:
                print(f"Error in file {error['instance']}: {error['reason']}")
                bad_instances.append(error)
    frame = pd.concat(good_data, ignore_index=True) if good_data else None
    return frame, bad_instances
                
                
                
                
                




def check_bin_lb_sum_div_c(old_df, filter=True):
    '''checks a results dataframes bin_lb column against max_div_c. it should not be less than sum_div_c. returns the dataframe with a column for the check'''
    df = old_df.copy()
    df['red_flag'] = df['bin_lb'] < df['sum_div_c']
    #prints the instances where the check column is True
    print("Warning, the following instances had bad bin_lb bounds,", df[df['red_flag']==True]['instance'].unique())
    if filter:
        #Filters the dataframe to only include the rows where the check column is False
        df = df[df['red_flag']==False]
        df = df.drop(columns=['red_flag'])
    return df


def check_min_bin_lb(old_df, filter=True):
    '''checks a results dataframes min column against bin lb. it should not be less than min. returns the dataframe with a column for the check'''
    df = old_df.copy()
    df['red_flag'] = df['min'] < df['bin_lb']
    #prints the instances where the check column is True
    print("Warning, the following instances had bad min bounds compared to bin_lb,", df[df['red_flag']==True]['instance'].unique())
    if filter:
        #Filters the dataframe to only include the rows where the check column is False
        df = df[df['red_flag']==False]
        df = df.drop(columns=['red_flag'])
    return df

def check_max_v_average(old_df):
    '''Checks the max vs the average number of stations accross instances'''
    df = old_df.copy()
    station_averages = df.groupby('instance').agg({'no_stations': 'mean'}).reset_index()
    station_averages.columns = ['instance', 'average_no_stations']
    #merges the average number of stations with the original dataframe
    df = pd.merge(df, station_averages, on='instance', how='left')
    df['yellow_flag'] = df['no_stations'] > df['average_no_stations'] + 1
    print("Warning, the following instances had bad max bounds compared to the average number of stations,", df[df['yellow_flag']==True]['instance'].unique())
    return df



def run_basic_checks(df, filter=True):
    '''Runs all the basic checks on the dataframe'''
    df = check_bin_lb_sum_div_c(df, filter=filter)
    df = check_min_bin_lb(df, filter=filter)
    df = check_max_v_average(df)
    return df

def check_n_edges(df, filter=True, edge_column ='n_edges'):
    '''Makes sure the df going to the GNN has the length of is_less_max equal to n_edges'''
    df[edge_column] = df[edge_column].astype(int)
    station_lengths = df['no_stations'].apply(len)
    df['red_flag'] = station_lengths != df[edge_column]
    print("Warning, the following instances are missing edge data,", df[df['red_flag']==True]['instance'].unique())
    if filter:
        print("Removing instances with missing edge data")
        df = df[df['red_flag']==False]
        df = df.drop(columns=['red_flag'])
    return df

def add_min_and_max(df, column):
    min_and_max = df.groupby("instance")[column].agg(["min", "max"])
    min_and_max.reset_index(inplace = True)
    # #merges min and max with the tasks_50 dataframe
    df = pd.merge(df, min_and_max, on = "instance")
    return df

def make_df_for_gnn(old_df):
    '''Prepares the data to be given to the GNN'''
    #First, we need to check the data
    df = old_df.copy()
    df = run_basic_checks(df)
    #prints the instances with yellow flags
    print("Removed red flags")
    print("The following instances had yellow flags,", df[df['yellow_flag']==True]['instance'].unique())
    df['is_less_max'] = df['no_stations'] < df['max']
    df['min_less_max'] = df['min'] < df['max']
    df = df.groupby('instance').agg({'no_stations': list, 'edge': list, 'is_less_max':list, 'min_less_max': 'first', 'n_edges':'first'}).reset_index()
    df = check_n_edges(df)
    return df

def prep_data_for_gnn(results_folder, graph_data_df_fp,collated_out, gnn_dat_out, ml_dat_out):
    ''' makes data appropiate for the DataSet used by the GNN. It needs the SALBP solver results, and graph meta data df before calculating the rest'''
    my_df, bad_instances = process_SALBP_res_folder(results_folder)
    my_df = my_df.loc[:, ~my_df.columns.str.contains("^Unnamed")]
    my_df.to_csv(collated_out, index=False)
    #my_df = pd.read_csv()
    graph_data = pd.read_csv(graph_data_df_fp)
    graph_data = graph_data.loc[:, ~graph_data.columns.str.contains("^Unnamed")]
    w_graph_attributes = pd.merge(my_df, graph_data, on = 'instance', how="left")
    w_graph_attributes = add_min_and_max(w_graph_attributes)
    
    w_graph_attributes.rename(columns={'nodes':'edge', 'original_n_precedence_constraints':'n_edges'}, inplace=True)
    w_graph_attributes.to_csv(ml_dat_out, index=False)
    gnn_df = make_df_for_gnn(w_graph_attributes)
    gnn_df.to_csv(gnn_dat_out, index=False)
    return gnn_df


def prep_data_for_gnn_2(result_csv, graph_data_df_fp, edge_data_df_fp, gnn_dat_out, ml_dat_out, remove_incomplete= True, tolerance=0, obj_col="no_stations", new_name="s_orig"):
    ''' makes data appropiate for the DataSet used by the GNN. It needs the SALBP solver results, and graph meta data df before calculating the rest'''
    my_df = pd.read_csv(result_csv)

    my_df = get_salbp_ub(my_df, obj_col=obj_col, new_name=new_name)
    print("res data columns after ub", my_df.columns)
    if remove_incomplete:
        instance_counts = my_df.groupby('instance')['precedence_relation'].count().reset_index()
        instance_counts.rename(columns={"precedence_relation":"row_counts"}, inplace=True)
        my_df = pd.merge(my_df, instance_counts, how="left")
        removed = my_df[my_df["row_counts"] +tolerance < my_df["original_n_precedence_constraints"]]
        print("removing: ", removed['instance'].unique())
        my_df = my_df[my_df["row_counts"] +tolerance >= my_df["original_n_precedence_constraints"]]
        my_df = my_df.drop(columns={'row_counts'})
    #my_df = pd.read_csv()
    graph_data = pd.read_csv(graph_data_df_fp)
    if 'n_stations' in graph_data.columns:
        graph_data.drop(columns=['n_stations'],inplace=True)
    print("graph_data cols", graph_data.columns)
    graph_data = graph_data.loc[:, ~graph_data.columns.str.contains("^Unnamed")]
    
    edge_data = pd.read_csv(edge_data_df_fp)
    print("edge data columns", edge_data.columns)
    edge_data = edge_data.loc[:, ~edge_data.columns.str.contains("^Unnamed")]
    graph_data = pd.merge(graph_data, edge_data, on="instance", how="right")
    my_df = pd.merge(my_df, graph_data, left_on = ['instance','precedence_relation'], right_on = ['instance','idx'], how="inner")
    print("here are the columns now, ", my_df.columns)
    my_df = add_min_and_max(my_df, obj_col)
    my_df = my_df.drop(columns='idx')
    my_df =my_df.rename(columns={'nodes':'edge', 'original_n_precedence_constraints':'n_edges'})
    my_df.to_csv(ml_dat_out, index=False)
    # my_df = make_df_for_gnn(my_df)

    #my_df.to_csv(gnn_dat_out, index=False)
    return my_df


def prep_data_for_ml(result_csv, edge_data_df_fp,  ml_dat_out, remove_incomplete= True, tolerance=0, obj_col="no_stations", new_name="s_orig"):
    ''' makes data appropiate for the DataSet used by the GNN. It needs the SALBP solver results, and graph meta data df before calculating the rest'''
    my_df = pd.read_csv(result_csv)

    my_df = get_salbp_ub(my_df, obj_col=obj_col, new_name=new_name)
    if remove_incomplete:
        instance_counts = my_df.groupby('instance')['precedence_relation'].count().reset_index()
        instance_counts.rename(columns={"precedence_relation":"row_counts"}, inplace=True)
        my_df = pd.merge(my_df, instance_counts, how="left")
        removed = my_df[my_df["row_counts"] +tolerance < my_df["original_n_precedence_constraints"]]
        print("removing: ", removed['instance'].unique())
        my_df = my_df[my_df["row_counts"] +tolerance >= my_df["original_n_precedence_constraints"]]
        my_df = my_df.drop(columns={'row_counts'})
    #my_df = pd.read_csv()

    edge_data = pd.read_csv(edge_data_df_fp)
    print("edge data columns", edge_data.columns)
    print('idx in columns', 'idx' in edge_data.columns, 'instance' in edge_data.columns)
    print('my_df', 'instance' in my_df.columns, 'precedence_relation' in my_df.columns)
    edge_data = edge_data.loc[:, ~edge_data.columns.str.contains("^Unnamed")]
    my_df['precedence_relation'] = my_df['precedence_relation'].astype('int64')
    edge_data['idx'] = edge_data['idx'].astype('int64')
    print('edge_data instances', edge_data['instance'])
    print('my_df ', my_df['instance'])
    my_df = pd.merge(my_df, edge_data, left_on = ['instance','precedence_relation'], right_on = ['instance','idx'], how="inner")
    print("here are the columns now, ", my_df.columns)
    print("dataframe after merge: ", my_df)
    my_df = add_min_and_max(my_df, obj_col)
    my_df = my_df.drop(columns='idx')
    my_df =my_df.rename(columns={'nodes':'edge', 'original_n_precedence_constraints':'n_edges'})
    my_df.to_csv(ml_dat_out, index=False)
    # my_df = make_df_for_gnn(my_df)

    #my_df.to_csv(gnn_dat_out, index=False)
    return my_df

def get_salbp_ub(my_df, obj_col="no_stations", new_name="s_orig"):
    my_df = my_df.loc[:, ~my_df.columns.str.contains("^Unnamed")]
    orig_res = my_df[my_df['nodes']=="SALBP_original"]
    orig_res = orig_res[['instance',obj_col]]
    if new_name:
        orig_res.rename(columns={obj_col:new_name},inplace=True)
    my_df = my_df[~(my_df['nodes']=="SALBP_original")]
    my_df = pd.merge(my_df, orig_res, how="left", on='instance' )
    return my_df



def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Calcuate graph metadata for an ALB instance')
    
    # Add arguments
    parser.add_argument('--pickle_fp', type=str, required=True, help='filepath for alb pickle dataset')
    parser.add_argument('--output_fp', type=str, required=True, help='filepath for results, if no error')
    parser.add_argument('--n_processes', type=int, required=False, default=1, help='Number of processes to use')
    parser.add_argument('--feature_type',type=str, required=True, help='generating graph or edge data')
    parser.add_argument('--cap_constraint', type=int, required=False, help='Cycle time for salb1, n_stations for salb2')
    parser.add_argument('--salbp', type=str, required=False,default= 'salbp_1', help='type of salbp problem we are solving' )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input
    if args.feature_type == 'graph':
        results = generate_graph_data_from_pickle( pickle_instance_fp = args.pickle_fp, pool_size=args.n_processes, salbp_type=args.salbp, cap_constraint=args.cap_constraint )
    elif args.feature_type == 'edge':
        results = generate_edge_data_from_pickle(pickle_instance_fp = args.pickle_fp, pool_size=args.n_processes )
    elif args.feature_type =='all':
        results = generate_all_features_from_pickle(pickle_instance_fp= args.pickle_fp, pool_size=args.n_processes, salbp_type=args.salbp, cap_constraint=args.cap_constraint)
    else:
        raise ValueError("Must select 'graph' or 'edge' for the type argument")
    # Process the range
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_fp, index=False)



if __name__ == "__main__":
    main()