import re
import networkx as nx
import sys
import os
import pandas as pd
# Add the folder containing your module to the Python path
build_dir = '/Users/letshopethisworks2/CLionProjects/SALBP_ILS/cmake-build-python_interface/'
sys.path.insert(0, build_dir)
build_dir_2 = '/home/jot240/DADA/SALBP_ILS/build/'
sys.path.insert(0, build_dir_2)
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
from pathlib import Path
import numpy as np

import re




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
    for line in lines:
        line = line.strip()
        
        # Look for the results line with verified_optimality, value, and cpu
        if "verified_optimality" in line and "value" in line and "cpu" in line:
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
                        #first_num = int(parts[0]) - 1  # Convert to 0-based index
                        second_num = int(parts[1])
                        solution_list.append( second_num)
                    except ValueError:
                        continue
    
    return {
        'verified_optimality': verified_optimality,
        'n_stations': value,
        'cpu': cpu,
        'task_assignments': solution_list
    }





class SALBP2Base:
    """Base class for SALBP-2 solvers"""
    
    def __init__(self, ex_fp, branch, total_time_limit=3600):
        self.ex_fp = ex_fp
        self.branch = branch
        self.total_tl = total_time_limit
    
    def _prepare_problem_data(self, salbp_dict):
        """Common preprocessing of problem data"""
        task_times = [task_time for (_, task_time) in salbp_dict['task_times'].items()]
        raw_precedence = [[int(parent), int(child)] for (parent, child) in salbp_dict['precedence_relations']]
        return task_times, raw_precedence
    
    def _finalize_solution(self, best_sol, n_stations, start_time, orig_lb, optimal=False):
        """Common solution finalization"""
        best_sol['verified_optimality'] = optimal
        best_sol['n_stations'] = n_stations
        best_sol['cpu'] = time.time() - start_time
        best_sol['bin_salbp2_lb'] = orig_lb
        return best_sol
    
    def solve(self, salbp_dict, n_stations, init_ub=None, init_sol=None):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement solve method")

class SALBP2WS(SALBP2Base):
    def __init__(self,ex_fp, branch, total_time_limit=3600, expected_iterations=6, vdls_time_limit=1,  use_vdls=True , iteration_time = None):
        super().__init__(ex_fp, branch, total_time_limit)
        self.ei = expected_iterations
        self.vdls_tl = vdls_time_limit
        self.use_vdls = use_vdls
        self.iteration_time = iteration_time


    def solve(self,salbp_dict, n_stations,init_ub=1e20, init_sol = []): 
        start = time.time()
        task_times , raw_precedence = self._prepare_problem_data(salbp_dict)
        lb = ils.calc_salbp_2_lbs(task_times, n_stations)
        orig_lb = lb
        if self.use_vdls:
            print("solving salbp2: using vdls for starting heuristic")
            if not init_sol:
                vdls_sol = ils.vdls_solve_salbp2(
                                                    S=n_stations,
                                                    N=salbp_dict['num_tasks'],
                                                    task_times=task_times,
                                                    raw_precedence=raw_precedence,
                                                    time_limit = self.vdls_tl,
                                                            )
            else:
                vdls_sol = ils.vdls_solve_salbp2(
                                                    S=n_stations,
                                                    N=salbp_dict['num_tasks'],
                                                    task_times=task_times,
                                                    raw_precedence=raw_precedence,
                                                    time_limit = self.vdls_tl,
                                                    initial_solution = init_sol
                                                            )
            ub = min(vdls_sol.cycle_time, init_ub)

            best_sol = {"cycle_time": ub, "task_assignments":  vdls_sol.task_assignment}
        else:
            ub = min(ils.calc_salbp_2_ub(task_times, n_stations), init_ub)
            task_assignments = None
            if init_sol:
                task_assignments = init_sol
            best_sol = {"cycle_time": ub, "task_assignments":  task_assignments}
        a0 = 0 #Fibonnacci search variables
        a1 = 1
        if not self.iteration_time:
            iteration_time = int((time.time()-start)/self.ei)

        if ub == lb:
            lb_optimal = True
            ub_optimal = True
        else:
            lb_optimal = False
            ub_optimal = False
            print("VDLS solution cycle time: ", ub, " lb ", lb,  " starting search with bbr")


        while lb < ub and not time_exceeded(start, self.total_tl):
            
            print("current lower bound", lb, "current upper bound", ub)
            test_salbp_dict = deepcopy(salbp_dict)
            test_salbp_dict["cycle_time"] = lb + a0
            results = salbp1_bbr_call(test_salbp_dict,self.ex_fp, self.branch, time_limit = iteration_time)
            if results["n_stations"] is not None and results["n_stations"] <= n_stations:
                ub = test_salbp_dict["cycle_time"]
                if results['verified_optimality'] == True:
                    ub_optimal = True
                best_sol = {"cycle_time": test_salbp_dict["cycle_time"], "task_assignments":  results["task_assignments"]}
                a0 = 0 #Restarts search at best lower bound
                a1 = 1

                
            else:
                lb =test_salbp_dict["cycle_time"]+1 #Did not get a valid solution, so the lower bound must at least be one larger
                if results['verified_optimality']:
                    lb_optimal = True
                a2 = a1 + a0 #Increases the step size for search
                a0 = a1
                a1 = a2
        optimal = False
        if ub_optimal and lb_optimal:
            optimal =True
        #Run again with remaining time 
        while not time_exceeded(start, self.total_tl) and not optimal:
            remaining_time =  self.total_tl - (time.time() - start)
            test_salbp_dict = deepcopy(salbp_dict)
            ub -=1
            test_salbp_dict["cycle_time"] = ub
            results = salbp1_bbr_call(test_salbp_dict,self.ex_fp, self.branch, time_limit = remaining_time)
            if results["n_stations"] is not None and results["n_stations"] <= n_stations:
                ub = test_salbp_dict["cycle_time"]
                best_sol = {"cycle_time": test_salbp_dict["cycle_time"], "task_assignments":  results["task_assignments"]}
            else: #Did not improve the cycle time
                if results['verified_optimality']:
                    optimal = True
                break
        best_sol = self._finalize_solution(best_sol, n_stations, start, orig_lb, optimal)
        return best_sol


def time_exceeded(start_time, time_limit):
    current_time = time.time()
    return current_time - start_time > time_limit

class SALBP2Li(SALBP2Base):
    def __init__(self,ex_fp,branch,total_time_limit = 3600,up_time_limit=10, down_time_limit=500, ):
        super().__init__(ex_fp, branch, total_time_limit)
        self.up_tl = up_time_limit
        self.down_tl = down_time_limit


    def solve(self,salbp_dict,  n_stations,init_ub=None, init_sol=[]): 
        '''Uses the search strategy from Enhanced branch-bound-remember and iterative beam search algorithms for type II assembly line balancing problem 2021'''
        start = time.time()
        task_times , _ = self._prepare_problem_data(salbp_dict)
        initial_ub_limit = min(int(np.ceil( self.total_tl / self.up_tl)), self.up_tl ) #Run initial search for at most 10 seconds, and at least 1
        lb = ils.calc_salbp_2_lbs(task_times, n_stations)
        optimal = True
        test_salbp_dict = deepcopy(salbp_dict)
        if not init_ub:
            #Search for ub solution using BBR
            ct = lb
            test_salbp_dict["cycle_time"] = ct
            results = salbp1_bbr_call(test_salbp_dict,self.ex_fp, self.branch, time_limit = initial_ub_limit)
            while results['n_stations'] > n_stations:
                ct +=1
                test_salbp_dict["cycle_time"] = ct
                results = salbp1_bbr_call(test_salbp_dict,self.ex_fp, self.branch, time_limit = initial_ub_limit)
            best_sol = {"cycle_time": test_salbp_dict["cycle_time"], "task_assignments":  results["task_assignments"]}
            optimal = False
            if (lb == ct):
                optimal = True
        else:
            print("using initial solution with ub: ", init_ub)
            best_sol = {"cycle_time": init_ub, "task_assignments": init_sol}
            ct = init_ub
        #start descending search
        ct -= 1
        while (lb < ct): 
            optimal=False
            remaining_limit = self.total_tl - (time.time() - start)
            if remaining_limit < 0:
                break
            solver_limit = min(remaining_limit, self.down_tl)
            test_salbp_dict["cycle_time"] = ct
            results = salbp1_bbr_call(test_salbp_dict,self.ex_fp, self.branch, time_limit = solver_limit)
            if results['n_stations'] <= n_stations:
                ct-=1
                best_sol = {"cycle_time": test_salbp_dict["cycle_time"], "task_assignments":  results["task_assignments"]}
            else:

                if results['verified_optimality']:
                    optimal = True
                print("Solution cycle time found", ct+1, " optimal: ", optimal)
                break

        best_sol = self._finalize_solution(best_sol, n_stations, start,lb, optimal )
        return best_sol
    
    


def salb2_solve(alb_dict,  out_fp,solver, n_stations):
    SALBP_dict_orig = alb_dict

    instance_fp = SALBP_dict_orig['name']
    results = []
    # Extract instance name from file path
    instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]
    out_path = Path(out_fp + '/'+ instance_name + ".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(out_path.name):
        orig_data = pd.DataFrame()    
    else:
        orig_data = pd.read_csv(out_path)
    print("running: ", instance_name, " saving to output ", out_fp)
    # Use a unique temporary ALB file per process
    
        
    orig_prec = len(SALBP_dict_orig["precedence_relations"])
    #original problem
    SALBP_dict = deepcopy(SALBP_dict_orig)
    if (orig_data.empty or orig_data[orig_data["nodes"] == "SALBP_original"].empty):
        
        result_dict =solver.solve(SALBP_dict, n_stations)
        # print("Return code:", output.returncode)
        # print("STDOUT:", output.stdout.decode())     
        metadata = {
            "instance": instance_name,
            "precedence_relation": "None",
            "nodes": "SALBP_original",
            "original_n_precedence_constraints": orig_prec,
        }
        result_dict = {**result_dict, **metadata}
        results.append(result_dict)
        save_backup(out_path, result_dict)
        #Tracking if instance autocompleted because bp=salbp and setting defaults
        cpu = -1 
        salbp2_sol = result_dict['cycle_time']
        bin_lb = result_dict['bin_salbp2_lb']
        task_assignments = result_dict['task_assignments']
    else:
        init_row = orig_data[orig_data["nodes"] == "SALBP_original"]
        bin_lb = init_row['bin_salbp2_lb'].iloc[0]
        salbp2_sol = init_row['cycle_time'].iloc[0]
        task_assignments = result_dict['task_assignments']
        cpu = -1 
        #TODO ADD OPTIMALITY CHECKS optimal = 1
    if not orig_data.empty:
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
        # if not orig_data.empty and not (orig_data[(orig_data["parent"] ==inst_par) & (orig_data["child"]==inst_chi)]).empty:
        #     print("Skipping relation that already exists: ", relation)
        #     result =        {
        #                         "cycle_time": salbp2_sol, 
        #                         "task_assignments":  task_assignments,
        #                         "cpu": cpu,
        #                         "n_stations": n_stations,
        #                         "bin_salbp2_lb":bin_lb

        #     }
        #     continue
        print("removing edge: ", relation)
        SALBP_dict = deepcopy(SALBP_dict_orig)
        SALBP_dict = precedence_removal(SALBP_dict, j)
        if bin_lb == salbp2_sol:
            print("original problem solution is already at lower bound, copying solution")
            result =         {    'cycle_time':salbp2_sol,
                              'task_assignments':task_assignments,
                                'verified_optimality':True,
                              'n_stations':n_stations,
                              'cpu': -1,
                              'bin_salbp2_lb':bin_lb,
                              



            }

        else:

            result =solver.solve(SALBP_dict,  n_stations,    salbp2_sol, init_sol = task_assignments)
        metadata = {
            "instance": instance_name,
            "precedence_relation": j,
            "nodes": relation,
            "original_n_precedence_constraints": orig_prec,
        }
        result = {**result, **metadata}
        save_backup(out_path, result)
        results.append(result)

    return results

def salbp2_prioirity_dict(alb_dict, n_random=100):
    S = alb_dict['n_stations']
    precs = alb_dict['precedence_relations']
    t_times = [val for _, val in alb_dict['task_times'].items()]
    N = len(t_times)
    precs = [[int(child), int(parent)]  for child, parent in alb_dict['precedence_relations']]
    start  = time.time()
    results = ils.priority_solve_salbp2(S=S, N=N, task_times= t_times, raw_precedence=precs, n_random=n_random )

    end = time.time()- start
    res_list = []
    for result in results:
        result_dict = {**result.to_dict(), "total_elapsed_time":end}
        res_list.append(result_dict)
    return res_list


def generate_salb_2_results_from_dict_list(alb_files, out_fp, solver,n_stations,  pool_size=4, ):
    with multiprocessing.Pool(pool_size) as pool:
        results = pool.starmap(salb2_solve, [(alb,  out_fp,solver, n_stations, ) for alb in alb_files])

    return results

def generate_salbp_2_results_from_pickle(fp  ,out_fp, solver, n_stations, res_df = None,    pool_size = 4, start=None, stop=None ):
    '''Solves SALBP instances. You can either pass an entire pickle file to try to solve all of its instances, 
    or an existing results dataframe with the pickle files to try to continue solving an existing dataset'''
    
    results = []
    #loads the pickle file

    if res_df is not None:

        res_df = pd.read_csv(res_df)
        if "pickle_fp" in res_df.columns:
            alb_files = []
            print("not using pickle fp")
            pickles = res_df['pickle_fp'].unique()
            for pf in pickles:
                alb_files +=  open_salbp_pickle(pf)
        else:
            alb_files = open_salbp_pickle(fp)
        print("here is the res df", res_df.head())
        instances = set(res_df['instance'])
        filtered_files = []
        for alb in alb_files:
            name = str(alb['name']).split('/')[-1].split('.')[0]
            if name not in instances:
                filtered_files.append(alb)
            else:
                print( name, "is already in results, skipping")


        results = generate_salb_2_results_from_dict_list(filtered_files,  out_fp, solver, n_stations,  pool_size)
    else:
        alb_files = open_salbp_pickle(fp)
        if start is not None and stop is not None:
            alb_files = alb_files[start:stop]
        results =  generate_salb_2_results_from_dict_list(alb_files, out_fp, solver, n_stations, pool_size)
    return results

def get_solver(solver_name, ex_fp, vdls_time_limit, branch, total_time_limit ):
    if solver_name == "li":
        solver = SALBP2Li(ex_fp, branch=branch, total_time_limit=total_time_limit)
    elif solver_name == "ws":
        if vdls_time_limit <= 0:
            solver = SALBP2WS(ex_fp, branch=branch, total_time_limit=total_time_limit, use_vdls=False )
        else:
            solver = SALBP2WS(ex_fp, branch=branch, total_time_limit=total_time_limit, vdls_time_limit=vdls_time_limit, )
    else:
        print("ERROR: no solver specified")
    return solver

def main():


    # Create argument parser
    parser = argparse.ArgumentParser(description='Solve edge removal on SALBP instance')
    
    # # # Add arguments
    parser.add_argument('--start', type=int, required=False, help='Starting integer (inclusive)')
    parser.add_argument('--end', type=int, required=False, help='Ending integer (inclusive)')
    parser.add_argument('--n_processes', type=int, required=False, default=1, help='Number of processes to use')
    parser.add_argument('--from_alb_folder', action="store_true", help='Whether to read albs directly from a folder, if false, reads from pickle')
    parser.add_argument('--SALBP_solver_fp', type=str, default="../BBR-for-SALBP1/SALB/SALB/salb", help='Filepath for SALBP solver')
    parser.add_argument('--filepath', type=str, required=True, help='filepath for alb dataset')
    parser.add_argument('--instance_name', type=str, required=False, help='start of instance name EX: "instance_n=50_"')
    parser.add_argument('--final_results_fp', type=str, required=True, help='filepath for results, if no error')
    parser.add_argument('--res_fp', type=str, required=False, help='Existing results df fp. Passing this filters out instances that have already been ran' )
    parser.add_argument('--solver_config', type=int, required=False, default=1, help='type of search strategy to use, 1 or 2 for the solver')
    parser.add_argument('--vdls_time', type=int, required=False, default=5, help='number of seconds for vdls initial solve (O for no vdls)')
    parser.add_argument('--n_stations', type=int, required=True, help='number of stations for salbp-2 station constraint' )
    parser.add_argument('--solver', type=str, required=False, default="li", help='what solver to use, li or ws' )
    parser.add_argument('--total_time_limit', type=int, required=False, default=3600, help="time limit for total solution")
    # # # Parse arguments
    args = parser.parse_args()
    Path(args.final_results_fp).mkdir(parents=True, exist_ok=True)
    solver = get_solver(args.solver, args.SALBP_solver_fp, vdls_time_limit = args.vdls_time, branch = args.solver_config, total_time_limit = args.total_time_limit )
    res  = generate_salbp_2_results_from_pickle(args.filepath, args.final_results_fp, solver=solver, n_stations=args.n_stations, res_df = args.res_fp,  pool_size=args.n_processes, start=args.start, stop=args.end)
    # Process t
    # # Validate input





if __name__ == "__main__":
    main()
#reads the results csv
#results_df = pd.read_csv("task_20_bin_lb.csv")
#results_df = pd.DataFrame(results)
#saves the results df to a csv file
#results_df.to_csv("tasks20_test.csv")
