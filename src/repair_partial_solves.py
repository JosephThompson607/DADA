import re
import networkx as nx
import matplotlib.pyplot as plt
import pydot
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
import re
import pandas as pd
import sys
import subprocess
from copy import deepcopy
from pathlib import Path
import os
from pathlib import Path
import pickle
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import tempfile
import glob
from alb_instance_compressor import parse_alb, write_to_alb, open_salbp_pickle
import subprocess
import pandas as pd
from copy import deepcopy
import ILS_ALBP as ils 
import time
import ast
from SALBP_solve import *

def clean_up_csv(csv_fp):
    my_df = pd.read_csv(csv_fp, on_bad_lines='skip')
    my_df = my_df[my_df["cpu"] > -1.1]#removes rows where the solver completely failed
    my_df = my_df.loc[:, ~my_df.columns.str.startswith('Unnamed')]
    my_df.drop_duplicates(subset="precedence_relation", keep="first",inplace=True)
    my_df.to_csv(csv_fp, index=False)
    
    
def process_results_bad_lb(path, forgive_missing=False, forgive_lb_missmatch=False):
    '''Processes results for when the bin pack lower bound bottom row is broken'''

    path = Path(path).expanduser()  # Expands '~'
    all_files = list(path.glob("*.csv"))  # Finds all .csv files
    li = []
    bad_instances = []
    for filename in all_files:
        print('processing', filename)
        clean_up_csv(filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        if df.empty:
            print(f"Error: {filename} was empty")
            bad_instances.append({"instance":filename, "reason":"empty df"})
            continue
#         bin_lb = df[df["nodes"] == "SALBP_original"].copy()
#         bin_lb = bin_lb[bin_lb["bin_lb"].isna()==False].copy()
#         print(bin_lb)
#         salb_upper = df[df["nodes"] == "SALBP_original"].copy()
#         if salb_upper.empty:
#             bad_instances.append({"instance":filename, "reason":"no_ub"})
#             print(f"Error: {filename} is missing SALBP upper bound")
#             continue
#         if bin_lb.empty:
#             bad_instances.append({"instance":filename, "reason":"no_lb"})
#             print(f"Error: {filename} is missing a lower bound")
#             continue

#         df = df[df["precedence_relation"].isna() == False]
#         bin_lb.rename(columns = {"nodes":"bin_lb"}, inplace=True)
#         bin_lb = bin_lb[["instance", "bin_lb"]]

#         df = pd.merge(df, bin_lb, on="instance")
        if any(df['bin_lb'].isna() == True):
            print(f"Error: {filename} wasn't able to calculate a lower bound")
            bad_instances.append({"instance":filename, "reason":"no_bin_lb"})
            continue
        if any(df["no_stations"] < df["bin_lb"].astype(int)):
            print(f"Error: {filename} has a lower bound mismatch")
            bad_instances.append({"instance":filename, "reason":"lower bound mismatch"})
            if forgive_lb_missmatch == True:
                li.append(df)
            continue
        if len(df.index) != df["original_n_precedence_constraints"].iloc[0]+1:
            print("Error: ", filename, " is missing rows, please check")
            bad_instances.append({"instance":filename, "reason":"missing_edge"})
            if forgive_missing ==True:
                li.append(df)
        else:
            li.append(df)
    if len(li)>0:
        frame = pd.concat(li, axis=0, ignore_index=True)
    else:
        frame= None
    return frame, bad_instances

def fix_partial_result(alb_dict, ex_fp, out_fp, branch=1):
    SALBP_dict_orig = alb_dict

    instance_fp = SALBP_dict_orig['name']
    results = []
    # Extract instance name from file path
    instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]

    if not os.path.exists(out_fp):
        os.makedirs(out_fp)
        orig_data = pd.DataFrame()    
    else:
        orig_data = pd.read_csv(out_fp)
    print("running: ", instance_name, " saving to output ", out_fp)
    # Use a unique temporary ALB file per process
    with tempfile.NamedTemporaryFile(suffix=".alb", delete=True) as temp_alb:
        temp_alb_path = temp_alb.name  # Path to temporary file
        orig_prec = len(SALBP_dict_orig["precedence_relations"])
        #original problem
        SALBP_dict = deepcopy(SALBP_dict_orig)
        if (orig_data.empty or orig_data[orig_data["nodes"] == "SALBP_original"].empty):
            write_to_alb(SALBP_dict, temp_alb_path)
            output = subprocess.run([ex_fp, "-m", f"{branch}", "-b", "1", temp_alb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # print("Return code:", output.returncode)
            # print("STDOUT:", output.stdout.decode())

            salbp_sol, optimal, cpu, bin_lb = parse_bb_salb1_out(output)
            if not bin_lb:
                print("STDERR:", output.stderr.decode() if output.stderr else "No stderr captured")
                print("ERROR, no bin_lb", output)
            orig_prob = {
                "instance": instance_name,
                "precedence_relation": "None",
                "nodes": "SALBP_original",
                "no_stations": salbp_sol,
                "original_n_precedence_constraints": orig_prec,
                "optimal": optimal,
                "cpu": cpu,
                "bin_lb": bin_lb
            }
            results.append(orig_prob)
            save_backup(out_fp, orig_prob)
            #Tracking if instance autocompleted because bp=salbp and setting defaults
            cpu = -1 
            no_stations = salbp_sol
        else:
            init_row = orig_data[orig_data["nodes"] == "SALBP_original"]
            bin_lb = init_row['bin_lb'].iloc[0]
            salbp_sol = init_row['no_stations'].iloc[0]
            cpu = -1 
            optimal = 1
            no_stations = salbp_sol

        #proceeds to precedence constraint removal, if bin_lb != no stations
        orig_data = orig_data[~(orig_data["nodes"]=="SALBP_original")]

        orig_data[["parent", "child"]] = orig_data["nodes"].apply(
                                                    lambda x: pd.Series([i for i in ast.literal_eval(x)])
                                                        )

        for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
            (inst_par, inst_chi) = relation 
            # print("inst par and inst chi", inst_par, inst_chi)
            # print(orig_data[orig_data["parent"] ==inst_par])
            # print(orig_data[orig_data["child"]==inst_chi])
            # print("parent", orig_data["parent"])
            # print("child", orig_data["child"])
            if not orig_data.empty and not (orig_data[(orig_data["parent"] ==inst_par) & (orig_data["child"]==inst_chi)]).empty:
                print("Skipping relation that already exists: ", relation)
                continue
            print("removing edge: ", relation)
            SALBP_dict = deepcopy(SALBP_dict_orig)
            SALBP_dict = precedence_removal(SALBP_dict, j)
            if bin_lb != salbp_sol: #If bin_lb==salbp_sol, then we don't need to do any precedence removal
                write_to_alb(SALBP_dict, temp_alb_path)
                output = subprocess.run([ex_fp, "-m", f"{branch}", temp_alb_path], stdout=subprocess.PIPE)
                # print("Return code:", output.returncode)
                # print("STDOUT:", output.stdout.decode())
                print("STDERR:", output.stderr.decode() if output.stderr else "No stderr captured")
                no_stations, optimal, cpu, _ = parse_bb_salb1_out(output)
            result = {
                "instance": instance_name,
                "precedence_relation": j,
                "nodes": relation,
                "no_stations": no_stations,
                "original_n_precedence_constraints": orig_prec,
                "optimal": optimal,
                "cpu": cpu,
                "bin_lb": bin_lb
            }
            save_backup(out_fp, result)
            results.append(result)

    return results

def retry_bad_instance(instance_fp, alb_dicts, ex_fp, branch):
    instance_name = instance_fp.split('/')[-1].split('.')[0]  
    clean_up_csv(instance_fp)
    for alb in alb_dicts:
        name = str(alb['name']).split('/')[-1].split('.')[0]
        if name == instance_name:
            print("fixing ", name , " at ", instance_fp)
            fix_partial_result(alb, ex_fp, instance_fp, branch=branch)


def retry_bad_instances(bad_data_fp, data_pickle, ex_fp,pool_size, branch):
    bad_data = pd.read_csv(bad_data_fp)
    alb_dicts = open_salbp_pickle(data_pickle)
    with multiprocessing.Pool(pool_size) as pool:
       pool.starmap(retry_bad_instance, [(instance, alb_dicts, ex_fp, branch) for instance in bad_data['instance']])



def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Solve edge removal on SALBP instance')
    
    # Add arguments
    parser.add_argument('--n_processes', type=int, required=False, default=1, help='Number of processes to use')
    parser.add_argument('--SALBP_solver_fp', type=str, default="../BBR-for-SALBP1/SALB/SALB/salb", help='Filepath for SALBP solver')
    parser.add_argument('--pkl_fp', type=str, required=True, help='filepath for alb dataset')
    parser.add_argument('--bad_data_csv', type=str, required=True, help='list of csv instance results instances that will be solved again')
    parser.add_argument('--solver_config', type=int, required=False, default=1, help='type of search strategy to use, 1 or 2 for the solver')
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input

    retry_bad_instances(args.bad_data_csv, args.pkl_fp, ex_fp=args.SALBP_solver_fp, pool_size=args.n_processes, branch = args.solver_config)
    # Process the range
    




if __name__ == "__main__":
    main()
#reads the results csv
#results_df = pd.read_csv("task_20_bin_lb.csv")
#results_df = pd.DataFrame(results)
#saves the results df to a csv file
#results_df.to_csv("tasks20_test.csv")
