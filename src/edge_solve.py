
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("../src"))
from alb_instance_compressor import *
import ast


from SALBP_solve import *
from copy import deepcopy
from metrics.node_and_edge_features import *
from metrics.graph_features import *
from metrics.time_metrics import *
#from data_prep import *
import multiprocessing
import time


def enumerate_list(edge_list):
    new_list = []
    for i,prec in enumerate(edge_list):
        new_prec = prec + [i]
        new_list.append(new_prec)
    return new_list

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
        
        
def make_missing_edge_df(bad_df, pkl_fp):
    print("starting")
    bad_df = bad_df[bad_df["reason"] == "missing_edge"]
    alb_df = create_pickle_df(pkl_fp)
    bad_inst, bad_fp = get_names_of_bad(bad_df)
    reduced_edges = alb_df[alb_df['instance_name'].isin(bad_inst)].copy()
    reduced_edges['save_fp'] = None
    for name, df_fp in zip(bad_inst,bad_fp):
        existing_results = pd.read_csv("../" + df_fp)
        existing_edges = existing_results['precedence_relation'].dropna()
        reduced_edges = reduced_edges[~((reduced_edges['instance_name'] == name) & (reduced_edges['edge_idx'].isin(existing_edges)))]
        reduced_edges.loc[reduced_edges['instance_name'] == name, 'save_fp'] = df_fp
        
    print("Following instances in dataframe:", reduced_edges['instance_name'].unique())

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