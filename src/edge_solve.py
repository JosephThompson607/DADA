
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("../src"))
from alb_instance_compressor import *
import ast
from datetime import datetime

from SALBP_solve import *
from copy import deepcopy
from metrics.node_and_edge_features import *
from metrics.graph_features import *
from metrics.time_metrics import *
#from data_prep import *
import multiprocessing
import time
from typing import Optional, Union




def get_names_of_bad(bad_df):
    names_of_bad = [ str(instance_fp).split("/")[-1].split(".")[0] for instance_fp in bad_df['instance']]
    fp_of_bad = [ str(instance_fp) for instance_fp in bad_df['instance']]
    return names_of_bad, fp_of_bad

def find_alb_instance(alb_dicts, instance_name):
    for idx, alb in enumerate(alb_dicts):
        name = str(alb['name']).split('/')[-1].split('.')[0]
        if name == instance_name:
            print("adding ", name , " at ", instance_fp)
            return idx, alb
    return -1, None


def find_missing_edges(existing_edges, alb_instance):
    missing_edges = []
    for idx, edge in alb_instance['precedence_constraint']:
        if idx not in existing_edges['precedence_constraint']:
            missing_edges.append({'idx':idx,'edge':edge})
    return missing_edges
        
        
def make_missing_edge_df(bad_df, pkl_fp, path = ""):
    if len(path)>0:
        os.chdir(path)
    print("starting")
    new_df = []
    bad_df = bad_df[bad_df["reason"] == "missing_edge"]
    alb_df = create_pickle_df(pkl_fp)
    bad_inst, bad_fp = get_names_of_bad(bad_df)
    reduced_edges = alb_df[alb_df['instance_name'].isin(bad_inst)].copy()
    reduced_edges['save_fp'] = None
    for name, df_fp in zip(bad_inst,bad_fp):
        existing_results = pd.read_csv( df_fp)
        existing_edges = existing_results['precedence_relation'].dropna()
        reduced_edges = reduced_edges[~((reduced_edges['instance_name'] == name) & (reduced_edges['edge_idx'].isin(existing_edges)))]
        reduced_edges.loc[reduced_edges['instance_name'] == name, 'save_fp'] = df_fp
        
    print("Following instances in dataframe:", reduced_edges['instance_name'].unique(), " there are ", len(reduced_edges), " rows")

    return reduced_edges

def solve_edge_salbp1(edge_df_fp, df_idx,ex_fp, branch =1):
    edge_df = pd.read_csv(edge_df_fp)
    edge = edge_df.iloc[df_idx]
    precedence_relations = [[int(x), int(y)] for x, y in ast.literal_eval(edge["precedence_relations"])]
    task_times = {k: int(v) for k, v in ast.literal_eval(edge['task_times']).items()}
    SALBP_dict = { #creating compatible instance format
    'num_tasks': edge['num_tasks'],
    'cycle_time': edge['cycle_time'], 
    'task_times': task_times,
    'precedence_relations': precedence_relations,
    'name': edge['name']
}
    with tempfile.NamedTemporaryFile(suffix=".alb", delete=True) as temp_alb:
        temp_alb_path = temp_alb.name  # Path to temporary file
        orig_prec = len(precedence_relations)
        #orig_data = pd.read_csv(edge['save_fp'])
        #init_row = orig_data[orig_data["nodes"] == "SALBP_original"]
        # bin_lb = init_row['bin_lb'].iloc[0]
        # salbp_sol = init_row['no_stations'].iloc[0]
        print(pd.isna(edge['precedence_relation']))
        if not pd.isna(edge['edge_idx']):
            SALBP_dict = precedence_removal(SALBP_dict, edge['edge_idx'])
        write_to_alb(SALBP_dict, temp_alb_path)
        output = subprocess.run([ex_fp, "-m", f"{branch}", temp_alb_path], stdout=subprocess.PIPE)
        # print("Return code:", output.returncode)
        # print("STDOUT:", output.stdout.decode())
        print("STDERR:", output.stderr.decode() if output.stderr else "No stderr captured")
        no_stations, optimal, cpu, _ = parse_bb_salb1_out(output)
        result = [{
            "instance": edge['instance_name'],
            "precedence_relation": edge['edge_idx'],
            "nodes": edge['precedence_relation'],
            "no_stations": no_stations,
            "original_n_precedence_constraints": orig_prec,
            "optimal": optimal,
            "cpu": cpu,
           "bin_lb": None
        }]
        print("THIS IS THE RESULT", result)
        p = Path(edge['save_fp'])
        new_folder = p.parent / p.stem
        new_folder.mkdir(exist_ok=True)
        # Save DataFrame to a new CSV in that folder
        new_csv_path = new_folder / Path(edge['instance_name'] +f"_pr_{str(edge['edge_idx'])}.csv")             
        res_df = pd.DataFrame(result)

        res_df.to_csv(new_csv_path, index=False)
        
        


def add_new_results(orig_res_fp, new_res_folder_fp):
    orig_res_df = pd.read_csv(orig_res_fp)
    new_res_df = concatenate_csvs_from_directory(new_res_folder_fp, add_source_column=True)
    sources = new_res_df['source_file']
    print("here are the sources", sources)
    new_res_df.drop(columns=['source_file'], inplace=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  

    new_res_df.to_csv(f"{new_res_folder_fp}collated_{timestamp}.res", index=False)
    fixed_results = merge_and_deduplicate_dataframes(new_res_df, orig_res_df)
    fixed_results.to_csv(orig_res_fp, index=False)
    for source_file in sources:
        if 'pr' in source_file:
            try:
                os.remove(new_res_folder_fp + "/" +source_file)
                print(f"Deleted: {source_file}")
            except FileNotFoundError:
                print(f"File not found: {source_file}")
            except Exception as e:
                print(f"Error deleting {source_file}: {e}")
    #return fixed_results

def merge_and_deduplicate_dataframes(
    new_df: pd.DataFrame,
    original_df: pd.DataFrame,
    no_stations_threshold: int = 0
) -> pd.DataFrame:
    """
    Process and merge two dataframes with specific filtering and deduplication rules.
    
    Parameters:
    -----------
    new_df : pd.DataFrame
        The new dataframe to be processed
    original_df : pd.DataFrame
        The original dataframe containing bin_lb values
    no_stations_threshold : int, default 0
        Threshold for filtering no_stations column (keeps rows >= threshold)
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with duplicates removed based on precedence_relation
    
    Raises:
    -------
    ValueError
        If required columns are missing from either dataframe
    """
    
    # Validate required columns
    if 'no_stations' not in new_df.columns:
        raise ValueError("'no_stations' column not found in new_df")
    
    if 'bin_lb' not in original_df.columns:
        raise ValueError("'bin_lb' column not found in original_df")
    
    if 'precedence_relation' not in new_df.columns and 'precedence_relation' not in original_df.columns:
        raise ValueError("'precedence_relation' column not found in either dataframe")
    
    # Step 1: Filter out rows where no_stations < threshold
    print(f"Original new_df shape: {new_df.shape}")
    filtered_new_df = new_df[new_df['no_stations'] >= no_stations_threshold].copy()
    print(f"After filtering no_stations >= {no_stations_threshold}: {filtered_new_df.shape}")
    
    # Step 2: Add bin_lb from original_df to new_df
    # If there's a way to match rows between dataframes, we'd need to know the key
    # For now, assuming we want to broadcast the bin_lb values from original_df
    if len(original_df['bin_lb'].unique()) == 1:
        # If original_df has a single bin_lb value, broadcast it
        filtered_new_df['bin_lb'] = original_df['bin_lb'].iloc[0]
        print(f"Added single bin_lb value: {original_df['bin_lb'].iloc[0]}")

    else:
        # If multiple values, we need a merge key - this would need clarification
        print("Warning: Multiple bin_lb values found. Using first value.")
        filtered_new_df['bin_lb'] = original_df['bin_lb'].iloc[0]

        
    # Step 3: Concatenate the dataframes
    print(f"Original_df shape: {original_df.shape}")
    combined_df = pd.concat([filtered_new_df, original_df], ignore_index=True, sort=False)
    print(f"Combined dataframe shape: {combined_df.shape}")
    
    # Step 4: Remove duplicates based on precedence_relation column
    initial_rows = len(combined_df)
    deduplicated_df = combined_df.drop_duplicates(subset=['precedence_relation'], keep='first')
    final_rows = len(deduplicated_df)
    
    print(f"Removed {initial_rows - final_rows} duplicate rows based on 'precedence_relation'")
    print(f"Final dataframe shape: {deduplicated_df.shape}")
    
    return deduplicated_df

def concatenate_csvs_from_directory(
    directory_path: Union[str, Path],
    ignore_index: bool = True,
    add_source_column: bool = False,
    file_pattern: str = "*.csv"
) -> pd.DataFrame:
    """
    Read all CSV files from a directory and concatenate them into a single DataFrame.
    
    Parameters:
    -----------
    directory_path : str or Path
        Path to the directory containing CSV files
    ignore_index : bool, default True
        If True, reset the index in the concatenated DataFrame
    add_source_column : bool, default False
        If True, add a column indicating the source file for each row
    file_pattern : str, default "*.csv"
        File pattern to match (e.g., "*.csv", "data_*.csv")
    
    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame from all CSV files
    
    Raises:
    -------
    ValueError
        If directory doesn't exist or no CSV files found
    """
    
    # Convert to Path object for easier handling
    dir_path = Path(directory_path)
    
    # Check if directory exists
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Find all CSV files matching the pattern
    csv_files = list(dir_path.glob(file_pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory_path}")
    
    # Read and store DataFrames
    dataframes = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Add source column if requested
            if add_source_column:
                df['source_file'] = csv_file.name
            
            dataframes.append(df)
            print(f"Successfully read: {csv_file.name} ({len(df)} rows)")
            
        except Exception as e:
            print(f"Warning: Could not read {csv_file.name}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No CSV files could be successfully read")
    
    # Concatenate all DataFrames
    result_df = pd.concat(dataframes, ignore_index=ignore_index, sort=False)
    
    print(f"\nConcatenation complete:")
    print(f"- Files processed: {len(dataframes)}")
    print(f"- Total rows: {len(result_df)}")
    print(f"- Total columns: {len(result_df.columns)}")
    
    return result_df

def file_path_to_dir_path(file_path):
    return os.path.splitext(file_path)[0] + "/"

def check_on_bad_instances(bad_csv_fp, path =""):
    if len(path)>0:
        os.chdir(path)
        print("currently at", os.getcwd())
    bad_instances = pd.read_csv(bad_csv_fp)
    bad_instances = bad_instances['instance']
    for instance in bad_instances:
        try:
            new_res_fp = file_path_to_dir_path(instance)
            print(new_res_fp)
            add_new_results(instance, new_res_fp)
        except FileNotFoundError as e:
            print(f"File not found: {new_res_fp} : {e}")
        except Exception as e:
            print(f"Error processing {new_res_fp}: {e}")
   
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Solve edge removal on SALBP instance')
    
    # Add arguments
    parser.add_argument('--SALBP_solver_fp', type=str, default="../BBR-for-SALBP1/SALB/SALB/salb", help='Filepath for SALBP solver')
    parser.add_argument('--idx', type=int, required=True, help='index of edge to solve on dataframe')
    parser.add_argument('--edge_data_csv', type=str, required=True, help='list of csv instance results instances that will be solved again')
    parser.add_argument('--solver_config', type=int, required=False, default=1, help='type of search strategy to use, 1 or 2 for the solver')
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input

    solve_edge_salbp1(args.edge_data_csv, args.idx, ex_fp=args.SALBP_solver_fp,branch = args.solver_config)
    # Process the range
    




if __name__ == "__main__":
    main()