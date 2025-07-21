import re
import networkx as nx
import sys
import os
import pandas as pd
# Add the folder containing your module to the Python path
build_dir = '/Users/letshopethisworks2/CLionProjects/SALBP_ILS/cmake-build-python_interface/'
sys.path.insert(0, build_dir)
sys.path.append('../src')
from alb_instance_compressor import *
import argparse
import re
import sys
import subprocess
from copy import deepcopy
import pickle
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import tempfile
import time
import glob
from alb_instance_compressor import parse_alb, write_to_alb, open_salbp_pickle
import subprocess
from copy import deepcopy
import ILS_ALBP as ils 
import ast
from SALBP_solve import *
from repair_partial_solves import *



def parse_alb_results2(file_content):
    """
    Parse ALB results file to extract:
    - verified_optimality
    - value
    - cpu
    - solution pairs as a list where first number is index-1 and second is value
    """
    
    lines = file_content.strip().split('\n')
    
    # Initialize results
    verified_optimality = None
    value = None
    cpu = None
    solution_list = []
    
    # Parse the file
    in_solution = False
    print("all lines:  " ,lines , "\n")
    for line in lines:
        line = line.strip()
        
        # Look for the results line with verified_optimality, value, and cpu
        if "verified_optimality" in line and "value" in line and "cpu" in line:
            print("this is the line: ",  line)
            # Parse the line: "verified_optimality = 1; value = 29; cpu = 10.17"
            parts = line.split(';')
            
            for part in parts:
                part = part.strip()
                if part.startswith('verified_optimality'):
                    verified_optimality = int(part.split('=')[1].strip())
                elif part.startswith('value'):
                    value = int(part.split('=')[1].strip())
                elif part.startswith('cpu'):
                    cpu = float(part.split('=')[1].strip())
        
        # Look for solution section
        if line.startswith("Solution with") and "stations" in line:
            in_solution = True
            continue
            
        # If we're in the solution section and hit a non-numeric line, we're done
        if in_solution:
            # Parse solution pairs
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) == 2:
                    try:
                        first_num = int(parts[0]) - 1  # Convert to 0-based index
                        second_num = int(parts[1])
                        solution_list.append([first_num, second_num])
                    except ValueError:
                        continue
    
    return {
        'verified_optimality': verified_optimality,
        'value': value,
        'cpu': cpu,
        'task_assignments': solution_list
    }


def salbp1_bbr_call(salbp_dict,ex_fp, branch):
    with tempfile.NamedTemporaryFile(suffix=".alb", delete=True) as temp_alb:
        temp_alb_path = temp_alb.name  # Path to temporary file
        write_to_alb(salbp_dict, temp_alb_path)
        output = subprocess.run([ex_fp, "-m", f"{branch}", "-b", "1", temp_alb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        results = parse_alb_results2(output.stdout.decode("utf-8"))
        print("results", results)
    return results


def bbr_salbp2(ex_fp,salbp_dict, branch, n_stations, vdls_time_limit=15):
    print("solving salbp2: using vdls for starting heuristic")
    start = time.time()
    task_times = [task_time for (_, task_time) in salbp_dict['task_times'].items() ]
    raw_precedence = [[int(parent), int(child)] for (parent, child) in salbp_dict['precedence_relations']]
    lb = ils.calc_salbp_2_lbs(task_times, n_stations)
    orig_lb = lb
    vdls_sol = ils.vdls_solve_salbp2(
    S=n_stations,
    N=salbp_dict['num_tasks'],
    task_times=task_times,
    raw_precedence=raw_precedence,
    time_limit = vdls_time_limit,
            )
    ub = vdls_sol.cycle_time
    print("VDLS solution cycle time: ", ub, " starting search with bbr")
    best_sol = {"cycle_time": ub, "task_assignments":  vdls_sol.task_assignment}
    a0 = 0
    a1 = 1
    while lb < ub:
        print("current lower bound", lb, "current upper bound", ub)
        test_salbp_dict = deepcopy(salbp_dict)
        test_salbp_dict["cycle_time"] = lb + a0
        results = salbp1_bbr_call(salbp_dict,ex_fp, branch)
        if results["value"] is not None and results["value"] <= n_stations:
            ub = test_salbp_dict["cycle_time"]
            best_sol = {"cycle_time": test_salbp_dict["cycle_time"], "task_assignments":  results["task_assignments"]}
            a0 = 0
            a1 = 1

            
        else:
            lb += a0
            a3 = a1 + a0
            a0 = a1
            a1 = a3
    best_sol['time'] = time.time() - start
    best_sol['bin_salbp2_lb'] = orig_lb
    return best_sol

   
    


# def salb2_solve(alb_dict, ex_fp, out_fp, branch=1):
#     SALBP_dict_orig = alb_dict

#     instance_fp = SALBP_dict_orig['name']
#     results = []
#     # Extract instance name from file path
#     instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]

#     if not os.path.exists(out_fp):
#         os.makedirs(out_fp)
#         orig_data = pd.DataFrame()    
#     else:
#         orig_data = pd.read_csv(out_fp)
#     print("running: ", instance_name, " saving to output ", out_fp)
#     # Use a unique temporary ALB file per process
    
        
#         orig_prec = len(SALBP_dict_orig["precedence_relations"])
#         #original problem
#         SALBP_dict = deepcopy(SALBP_dict_orig)
#         if (orig_data.empty or orig_data[orig_data["nodes"] == "SALBP_original"].empty):
            
#             output = subprocess.run([ex_fp, "-m", f"{branch}", "-b", "1", temp_alb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             # print("Return code:", output.returncode)
#             # print("STDOUT:", output.stdout.decode())

           

#             orig_prob = {
#                 "instance": instance_name,
#                 "precedence_relation": "None",
#                 "nodes": "SALBP_original",
#                 "no_stations": salbp_sol,
#                 "original_n_precedence_constraints": orig_prec,
#                 "optimal": optimal,
#                 "cpu": cpu,
#                 "bin_lb": bin_lb
#             }
#             results.append(orig_prob)
#             save_backup(out_fp, orig_prob)
#             #Tracking if instance autocompleted because bp=salbp and setting defaults
#             cpu = -1 
#             no_stations = salbp_sol
#         else:
#             init_row = orig_data[orig_data["nodes"] == "SALBP_original"]
#             bin_lb = init_row['bin_lb'].iloc[0]
#             salbp_sol = init_row['no_stations'].iloc[0]
#             cpu = -1 
#             optimal = 1
#             no_stations = salbp_sol

#         #proceeds to precedence constraint removal, if bin_lb != no stations
#         orig_data = orig_data[~(orig_data["nodes"]=="SALBP_original")]

#         orig_data[["parent", "child"]] = orig_data["nodes"].apply(
#                                                     lambda x: pd.Series([i for i in ast.literal_eval(x)])
#                                                         )

#         for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
#             (inst_par, inst_chi) = relation 
#             # print("inst par and inst chi", inst_par, inst_chi)
#             # print(orig_data[orig_data["parent"] ==inst_par])
#             # print(orig_data[orig_data["child"]==inst_chi])
#             # print("parent", orig_data["parent"])
#             # print("child", orig_data["child"])
#             if not orig_data.empty and not (orig_data[(orig_data["parent"] ==inst_par) & (orig_data["child"]==inst_chi)]).empty:
#                 print("Skipping relation that already exists: ", relation)
#                 continue
#             print("removing edge: ", relation)
#             SALBP_dict = deepcopy(SALBP_dict_orig)
#             SALBP_dict = precedence_removal(SALBP_dict, j)
#             if bin_lb != salbp_sol: #If bin_lb==salbp_sol, then we don't need to do any precedence removal
#                 write_to_alb(SALBP_dict, temp_alb_path)
#                 output = subprocess.run([ex_fp, "-m", f"{branch}", temp_alb_path], stdout=subprocess.PIPE)
#                 # print("Return code:", output.returncode)
#                 # print("STDOUT:", output.stdout.decode())
#                 print("STDERR:", output.stderr.decode() if output.stderr else "No stderr captured")
#                 no_stations, optimal, cpu, _ = parse_bb_salb1_out(output)
#             result = {
#                 "instance": instance_name,
#                 "precedence_relation": j,
#                 "nodes": relation,
#                 "no_stations": no_stations,
#                 "original_n_precedence_constraints": orig_prec,
#                 "optimal": optimal,
#                 "cpu": cpu,
#                 "bin_lb": bin_lb
#             }
#             save_backup(out_fp, result)
#             results.append(result)

#     return results

# def retry_bad_instance(instance_fp, alb_dicts, ex_fp, branch):
#     instance_name = instance_fp.split('/')[-1].split('.')[0]  
#     clean_up_csv(instance_fp)
#     for alb in alb_dicts:
#         name = str(alb['name']).split('/')[-1].split('.')[0]
#         if name == instance_name:
#             print("fixing ", name , " at ", instance_fp)
#             fix_partial_result(alb, ex_fp, instance_fp, branch=branch)


# def retry_bad_instances(bad_data_fp, data_pickle, ex_fp,pool_size, branch):
#     bad_data = pd.read_csv(bad_data_fp)
#     alb_dicts = open_salbp_pickle(data_pickle)
#     with multiprocessing.Pool(pool_size) as pool:
#        pool.starmap(retry_bad_instance, [(instance, alb_dicts, ex_fp, branch) for instance in bad_data['instance']])



def main():
    test_albp = parse_alb('test.alb')
    result = bbr_salbp2("../BBR-for-SALBP1/SALB/SALB/salb",test_albp, 1, 20, vdls_time_limit=15)
    print(result)

    # Create argument parser
    # parser = argparse.ArgumentParser(description='Solve edge removal on SALBP instance')
    
    # # Add arguments
    # parser.add_argument('--n_processes', type=int, required=False, default=1, help='Number of processes to use')
    # parser.add_argument('--SALBP_solver_fp', type=str, default="../BBR-for-SALBP1/SALB/SALB/salb", help='Filepath for SALBP solver')
    # parser.add_argument('--pkl_fp', type=str, required=True, help='filepath for alb dataset')
    # parser.add_argument('--bad_data_csv', type=str, required=True, help='list of csv instance results instances that will be solved again')
    # parser.add_argument('--solver_config', type=int, required=False, default=1, help='type of search strategy to use, 1 or 2 for the solver')
    # # Parse arguments
    # args = parser.parse_args()
    
    # # Validate input

    # retry_bad_instances(args.bad_data_csv, args.pkl_fp, ex_fp=args.SALBP_solver_fp, pool_size=args.n_processes, branch = args.solver_config)
    # Process the range
    




if __name__ == "__main__":
    main()
#reads the results csv
#results_df = pd.read_csv("task_20_bin_lb.csv")
#results_df = pd.DataFrame(results)
#saves the results df to a csv file
#results_df.to_csv("tasks20_test.csv")
