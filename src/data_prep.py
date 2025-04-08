import glob
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from alb_instance_compressor import *
from copy import deepcopy
from metrics.node_and_edge_features import *
from metrics.graph_features import *
from metrics.time_metrics import *
import multiprocessing


#FEATURE GENERATION FROM GRAPH
def alb_to_edge_data(alb_instance):
    instance = str(alb_instance['name']).split('/')[-1].split('.')[0]
    print("processing instance", instance)
    alb_dag = make_alb_digraph(alb_instance)
    time_metrics = get_time_stats(alb_instance, C=1000)
 
    graph_metrics = get_graph_metrics(alb_instance)
    edge_data = {'instance':instance, **time_metrics, **graph_metrics}
    return edge_data



def generate_edge_data_from_pickle(pickle_instance_fp = "raw/pkl_datasets/n_50_bottleneck.pkl", pool_size=4 ):
    alb_instances = open_salbp_pickle(pickle_instance_fp)
    with multiprocessing.Pool(pool_size) as pool:
        edge_data = pool.map(alb_to_edge_data, alb_instances)
    print(edge_data)
    edge_info_df = pd.DataFrame(edge_data)
    return edge_info_df


#PROCESSING SALBP OUTPUT
def process_file(filename):
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
        if len(df.index) != df["original_n_precedence_constraints"].iloc[0]:
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
    print(df.columns)
    df['yellow_flag'] = df['no_stations'] > df['average_no_stations'] + 1
    print("Warning, the following instances had bad max bounds compared to the average number of stations,", df[df['yellow_flag']==True]['instance'].unique())
    return df



def run_basic_checks(df, filter=True):
    '''Runs all the basic checks on the dataframe'''
    df = check_bin_lb_sum_div_c(df, filter=filter)
    df = check_min_bin_lb(df, filter=filter)
    df = check_max_v_average(df)
    return df

def check_n_edges(df, filter=True):
    '''Makes sure the df going to the GNN has the length of is_less_max equal to n_edges'''
    df['n_edges'] = df['n_edges'].astype(int)
    station_lengths = df['no_stations'].apply(len)
    df['red_flag'] = station_lengths != df['n_edges']
    print("Warning, the following instances are missing edge data,", df[df['red_flag']==True]['instance'].unique())
    if filter:
        print("Removing instances with missing edge data")
        df = df[df['red_flag']==False]
        df = df.drop(columns=['red_flag'])
    return df

def add_min_and_max(df):
    min_and_max = df.groupby("instance")["no_stations"].agg(["min", "max"])
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


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Solve edge removal on SALBP instance')
    
    # Add arguments
    parser.add_argument('--filepath', type=str, required=True, help='filepath for alb pickle dataset')
    parser.add_argument('--output_fp', type=str, required=True, help='filepath for results, if no error')
    parser.add_argument('--n_processes', type=int, required=False, default=1, help='Number of processes to use')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input

    results = generate_edge_data_from_pickle( pickle_instance_fp = args.filepath, pool_size=args.n_processes )
    # Process the range
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_fp, index=False)



if __name__ == "__main__":
    main()