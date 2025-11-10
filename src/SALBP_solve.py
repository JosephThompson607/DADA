import re
import networkx as nx
import matplotlib.pyplot as plt
import pydot
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import yaml
import argparse
import re
import pandas as pd
import sys
import subprocess
from copy import deepcopy
build_dir = '/Users/letshopethisworks2/CLionProjects/SALBP_ILS/cmake-build-python_interface/'
sys.path.insert(0, build_dir)
build_dir_2 = '/home/jot240/DADA/SALBP_ILS/build/'
sys.path.insert(0, build_dir_2)
from pathlib import Path
import os
from pathlib import Path
import pickle
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import tempfile
import glob
from alb_instance_compressor import parse_alb, write_to_alb, open_salbp_pickle, get_instance_name
import subprocess
import pandas as pd
from copy import deepcopy
import ILS_ALBP as ils 
import time
from datetime import date
import shutil
def salbp1_bbr_call(salbp_dict,ex_fp, branch=1, time_limit=3600, w_bin_pack = True,orig_bbr=True, **kwargs):
    start = time.time()
    with tempfile.NamedTemporaryFile(suffix=".alb", delete=True) as temp_alb:
        temp_alb_path = temp_alb.name  # Path to temporary file
        write_to_alb(salbp_dict, temp_alb_path)
        #output = subprocess.run([ex_fp, "-m", f"{branch}", "-b", "1", temp_alb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if w_bin_pack:
            output = subprocess.run([ex_fp, "-m", f"{branch}", "-b", "1", "-t", f"{time_limit}", temp_alb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            output = subprocess.run([ex_fp, "-m", f"{branch}","-t", f"{time_limit}", temp_alb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #print("probelm solved. Parsing. Time: ", time.time()-start)
        if orig_bbr:
            results = parse_alb_results_orig_bbr(output.stdout.decode("utf-8"))
        else:
            results = parse_alb_results_new_bbr(output.stdout.decode("utf-8"))
    return results

def random_task_time_change(SALBP_dict, multiplier = 1.5):
    """Increases a random task time by 1"""
    import random
    task_id = random.choice(list(SALBP_dict["task_times"].keys()))
    SALBP_dict["task_times"][task_id] *= multiplier
    return SALBP_dict

def task_time_change(SALBP_dict, task_id, multiplier = 1.5, debug = False):
    """Increases a random task time by 1"""
    if debug:
        print("Changing task", task_id, "time by", multiplier)
    SALBP_dict["task_times"][task_id] *= multiplier
    return SALBP_dict

def precedence_removal(SALBP_dict, edge_index):
    """Removes a precedence relation"""
    SALBP_dict["precedence_relations"].pop(edge_index)
    return SALBP_dict

def parse_alb_results_orig_bbr(output_text):
    """
    Parse ALB solver output and return results dictionary.

    Extracts: verified_optimality, value, cpu, bin_lb, task_assignments
    """
    result = {
        'verified_optimality': 0,
        'n_stations': None,
        'cpu': None,
        'bin_lb': None,
        'task_assignments': []
    }

    lines = output_text.strip().split('\n')
    in_solution = False
    pending_lbs = {}
    for line in lines:
        line = line.strip()

        # Enter or exit task assignment section
        if line.startswith("Solution with"):
            in_solution = True
            continue
        if in_solution and (not line or line.startswith("test")):
            in_solution = False

        # Parse task assignments: "task\tstation"
        if in_solution and re.match(r'^\d+\s+\d+', line):
            parts = line.split()
            result['task_assignments'].append(int(parts[1]))
        #lower bound search 

        if m := re.search(r'First lower bound:\s*(\d+)', line):
            pending_lbs['first'] = int(m.group(1))

        elif m := re.search(r'Second lower bound\s*(\d+)', line):
            pending_lbs['second'] = int(m.group(1))
        elif m := re.search(r'Bin-?packing lower bound\s*:?[\s]*(\d+)', line):
            result['bin_lb'] = int(m.group(1))
        # Parse verified_optimality, value, cpu
        if "verified_optimality" in line:
            if m := re.search(r'verified_optimality\s*=\s*(\d+)', line):
                result['verified_optimality'] = int(m.group(1))
            if m := re.search(r'value\s*=\s*(\d+)', line):
                result['n_stations'] = int(m.group(1))
            if m := re.search(r'cpu\s*=\s*([\d.]+)', line):
                result['cpu'] = float(m.group(1))
        # Parse bin packing lower bound
        if result['bin_lb'] is None and result['n_stations'] is not None:
            for key in ('first', 'second'):
                if key in pending_lbs and pending_lbs[key] == result['n_stations']:
                    result['bin_lb'] = pending_lbs[key]
                    break

    return result

def parse_alb_results_new_bbr(output_text):
    """
    Parse ALB solver output and return results dictionary.
    
    Args:
        output_text: String output from subprocess
        
    Returns:
        Dictionary with keys: verified_optimality, value, cpu, task_assignments
    """
    lines = output_text.strip().split('\n')
    
    # Initialize result dictionary
    result = {
        'verified_optimality': 0,
        'value': None,
        'cpu': None,
        'bin_lb':None,
        'task_assignments': []
    }
    
    # Parse task assignments
    in_task_assignments = False
    
    for line in lines:
        line = line.strip()
        
        # Check if we're entering task assignments section
        if line == '<task assignments>':
            in_task_assignments = True
            continue
        
        # Check if we're leaving task assignments section
        if line == '<task sequence>' or (in_task_assignments and line.startswith('<')):
            in_task_assignments = False
            continue
        
        # Parse task assignments (format: "task_number    station_number")
        if in_task_assignments and line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    task_num = int(parts[0])
                    station = int(parts[1])

                    result['task_assignments'].append(station)
                except ValueError:
                    continue
        
        # Parse the metrics line (contains "cpu:")
        if 'cpu:' in line:
            # Extract verified_optimality
            verified_match = re.search(r'verified_optimality\s+(\d+)', line)
            if verified_match:
                result['verified_optimality'] = int(verified_match.group(1))
            
            # Extract UB as the value
            ub_match = re.search(r'UB:\s*(\d+)', line)
            if ub_match:
                result['n_stations'] = int(ub_match.group(1))
               # Extract UB as the value
            bpp_match = re.search(r'bpp:\s*(\d+)', line)
            if bpp_match:
                result['bin_lb'] = int(bpp_match.group(1))
            
            # Extract CPU time
            cpu_match = re.search(r'cpu:\s*([\d.]+)', line)
            if cpu_match:
                result['cpu'] = float(cpu_match.group(1))
    return result

def parse_bb_salb1_out(text):
    '''gets the number of stations, optimal flag and cpu time from the output of the salb1 program'''
    output = text.stdout.decode("utf-8")
    lb_1 = re.search(r"First lower bound: (\d+)", output)
    bin_lb = re.search(r"Bin-packing lower bound (\d+)", output)
    if bin_lb:
        lb = int(bin_lb.group(1))
    elif lb_1:       #lb_1 is ceil(#station/C), also valid bin lb
        lb = int(lb_1.group(1))
    else:
        lb = None
    #search for final solutions
    match = re.search(r"verified_optimality\s*=\s*(\d+);\s*value\s*=\s*(\d+);\s*cpu\s*=\s*([\d.]+)", output)

    if match:
        verified_optimality = int(match.group(1))
        value = int(match.group(2))
        cpu = float(match.group(3))

        print("verified_optimality:", verified_optimality, " value: ", value," cpu: ", cpu)
    else:
        print("Pattern not found. output: ", output)
        value = -1000
        verified_optimality = 0
        cpu = -1000
    return value, verified_optimality, cpu, lb





def plot_salbp_graph(SALBP_dict):
    G = nx.DiGraph()
    G.add_nodes_from(SALBP_dict["task_times"].keys())
    G.add_edges_from(SALBP_dict["precedence_relations"])
    #prints the edges
    print("from dict", SALBP_dict["precedence_relations"])
    #prints the edges from the graph
    print("from graph", G.edges())
    nx.draw(G, with_labels = True)
    plt.show()

def plot_salbp_edge_removal_graph(SALBP_dict, instance_name, res_df):
    '''Colors the edges by the number of stations in res_df'''
    G = nx.DiGraph()
    G.add_nodes_from(SALBP_dict["task_times"].keys())
    G.add_edges_from(SALBP_dict["precedence_relations"])
    edge_colors = []
    for edge in G.edges():
        edge_index = SALBP_dict["precedence_relations"].index(list(edge))
        no_stations = res_df[(res_df["instance"] == instance_name) & (res_df["precedence_relation"] == edge_index)]["no_stations"].values[0]
        edge_colors.append(no_stations)
    #saves edge colors as graph attribute
    nx.set_edge_attributes(G, dict(zip(G.edges(), edge_colors)), "value")
    pos = nx.nx_pydot.graphviz_layout(G, prog = "dot")
   # Define colormap
    unique_values = list(set(edge_colors))
    print(unique_values)
    color_map = cm.get_cmap('viridis', len(unique_values))
    print("color map", color_map)
    cmap = mcolors.ListedColormap([color_map(val) for val in unique_values])

    # Draw graph
    #creates ax
    fig, ax = plt.subplots()
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=min(edge_colors), edge_vmax=max(edge_colors), ax=ax)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Add colorbar
    handles = [plt.Line2D([0], [0], marker='o', color = color_map(val), label=val, markersize=10) for val in unique_values]
    plt.legend(handles=handles, loc="best")

    plt.show()
    return G


def draw_graph_with_discrete_legend(SALBP_dict, res_df, instance_name,  ax=None, instance_key = 'instance'):
    G = nx.DiGraph()
    G.add_nodes_from(SALBP_dict["task_times"].keys())
    G.add_edges_from(SALBP_dict["precedence_relations"])

    edge_colors = []
    edge_values = []  # Store unique edge values for legend

    for edge in G.edges():
        edge_index = SALBP_dict["precedence_relations"].index(list(edge))
        no_stations = res_df[(res_df[instance_key] == instance_name) & 
                             (res_df["precedence_relation"] == edge_index)]["no_stations"].values[0]
        edge_colors.append(no_stations)
        if no_stations not in edge_values:
            edge_values.append(no_stations)

    # Save edge colors as graph attribute
    nx.set_edge_attributes(G, dict(zip(G.edges(), edge_colors)), "value")

    # Graph layout
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

    # Define discrete colormap
    unique_values = sorted(edge_values)
    num_colors = len(unique_values)
    cmap = plt.cm.get_cmap("Set1", num_colors)  # Use a qualitative colormap
    color_map = {val: cmap(i) for i, val in enumerate(unique_values)}  # Assign colors to unique values

    # Assign discrete colors to edges
    edge_color_list = [color_map[val] for val in edge_colors]

    # Draw graph
    if ax is None:
        fig, ax = plt.subplots()
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_color_list, ax=ax)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Create legend
    handles = [plt.Line2D([0], [0], color=color_map[val], lw=2, label=f"No. of Stations: {val}") for val in unique_values]
    #ax.legend(handles=handles, loc="best")


    return G



def save_backup(backup_name, result):
    intermediate = pd.DataFrame([result])
    my_file = Path(backup_name)
    if my_file.is_file():
        intermediate.to_csv(backup_name, mode='a', header=False, index=False)
    else:
        intermediate.to_csv(backup_name, index=False)
    
def generate_results(fp = "/Users/letshopethisworks2/Documents/phd_paper_material/MMABPWW/SALBP_benchmark/small data set_n=20/" ,  ex_fp = "../BBR-for-SALBP1/SALB/SALB/salb", instance_name = "instance_n=20_", ext = ".alb", start=1, stop = 300, backup_name = f"SALBP_edge_solutions.csv"):
    results = []
    for i in range(start,stop):
        SALBP_dict_orig = parse_alb(f"{fp}{instance_name}{i}{ext}")
        bin_dict = deepcopy(SALBP_dict_orig)
        print("instance name", instance_name, i)
        for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
            SALBP_dict = deepcopy(SALBP_dict_orig)
            SALBP_dict =precedence_removal(SALBP_dict, j)
            write_to_alb(SALBP_dict, "test.alb")
            output = subprocess.run([ex_fp,"-m","2", "test.alb"], stdout=subprocess.PIPE)
            no_stations, optimal, cpu = parse_bb_salb1_out(output)
            result = {"instance:": f"{instance_name}{i}", "precedence_relation": j, "nodes": relation,  "no_stations": no_stations, "optimal": optimal, "cpu": cpu}
            save_backup(backup_name, result)
            results.append(result)

        #calculates bin packing lower bound
        bin_dict['precedence_relations'] = []
        write_to_alb(bin_dict, "test.alb")
        #TODO ad -m 2 to hopefully avoid bug in solver code
        output = subprocess.run([ex_fp, "-m","2", "test.alb"], stdout=subprocess.PIPE)
        no_stations, optimal, cpu = parse_bb_salb1_out(output)
        result = {"instance": f"{instance_name}{i}", "precedence_relation": "None", "no_stations": no_stations, "optimal": optimal, "cpu": cpu}
        save_backup(backup_name, result)
            
        results.append(result)
    return results

def run_simple_alb_ils(inst):
    '''Runs ils on a .alb file'''
    
    alb = parse_alb(inst)
    alb['instance'] = inst.split("/")[-1].split(".")[0]
    results = run_alb_ils_dict(alb)
    return 


def run_alb_ils_dict(alb_dict, max_iterations):
    C = alb_dict['cycle_time']
    precs = alb_dict['precedence_relations']
    t_times = [val for _, val in alb_dict['task_times'].items()]
    precs = [[int(child), int(parent)]  for child, parent in alb_dict['precedence_relations']]
    start  = time.time()
    results = ils_call(cycle_time=C, tasks_times_list= t_times, precedence_list=precs, max_iterations=max_iterations)
    end = time.time()- start

    return {
            "instance":alb_dict['instance'],
            "n_stations": results.n_stations,
            "run_time": end}




# def generate_one_instance_results(alb_dict, ex_fp, out_fp, branch, time_limit):
#     SALBP_dict_orig = alb_dict
#     bin_dict = deepcopy(SALBP_dict_orig)
#     instance_fp = SALBP_dict_orig['name']
#     results = []
#     # Extract instance name from file path
#     instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]

#     if not os.path.exists(out_fp):
#          os.makedirs(out_fp)
#     print("running: ", instance_name, " saving to output ", out_fp, " time limit: ", time_limit)
#     # Use a unique temporary ALB file per process
   
#     orig_prec = len(SALBP_dict_orig["precedence_relations"])
#     #original problem
#     SALBP_dict = deepcopy(SALBP_dict_orig)
#     res = salbp1_bbr_call(SALBP_dict, ex_fp, branch, time_limit)
#     bin_lb = sum
#     salbp_sol = res['value']
#     optimal = res['verified_optimality']
#     cpu = res['cpu']
#     if not bin_lb:
#         print("ERROR, no bin_lb", )
#     orig_prob = {
#         "instance": instance_name,
#         "precedence_relation": "None",
#         "nodes": "SALBP_original",
#         "no_stations": salbp_sol,
#         "original_n_precedence_constraints": orig_prec,
#         "optimal": optimal,
#         "cpu": cpu,
#         "lb_1": bin_lb
#     }
#     results.append(orig_prob)
#     save_backup(out_fp+instance_name + ".csv", orig_prob)
#     #Tracking if instance autocompleted because bp=salbp and setting defaults
#     cpu = -1 
#     no_stations = salbp_sol

#     #proceeds to precedence constraint removal, if bin_lb != no stations
#     for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
#         print("removing edge: ", relation)
#         SALBP_dict = deepcopy(SALBP_dict_orig)
#         SALBP_dict = precedence_removal(SALBP_dict, j)
#         if bin_lb != salbp_sol: #If bin_lb==salbp_sol, then we don't need to do any precedence removal
#             res = salbp1_bbr_call(SALBP_dict, ex_fp, branch, time_limit)
#             bin_lb = res['bin_lb']
#             salbp_sol = res['value']
#             optimal = res['verified_optimality']
#             cpu = res['cpu']

#         result = {
#             "instance": instance_name,
#             "precedence_relation": j,
#             "nodes": relation,
#             "no_stations": no_stations,
#             "original_n_precedence_constraints": orig_prec,
#             "optimal": optimal,
#             "cpu": cpu,
#             "lb_1": bin_lb
#         }
#         save_backup(out_fp+instance_name + ".csv", result)
#         results.append(result)

#     return results
def generate_one_instance_results(alb_dict, ex_fp, out_fp, branch, time_limit):
    SALBP_dict_orig = alb_dict
    bin_dict = deepcopy(SALBP_dict_orig)
    instance_fp = SALBP_dict_orig['name']
    results = []
    # Extract instance name from file path
    instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]

    if not os.path.exists(out_fp):
         os.makedirs(out_fp)
    print("running: ", instance_name, " saving to output ", out_fp, " time limit: ", time_limit)
    # Use a unique temporary ALB file per process
    with tempfile.NamedTemporaryFile(suffix=".alb", delete=True) as temp_alb:
        temp_alb_path = temp_alb.name  # Path to temporary file
        orig_prec = len(SALBP_dict_orig["precedence_relations"])
        #original problem
        SALBP_dict = deepcopy(SALBP_dict_orig)
        write_to_alb(SALBP_dict, temp_alb_path)
        output = subprocess.run([ex_fp, "-m", f"{branch}", "-b", "1", "-t", f"{time_limit}", temp_alb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        save_backup(out_fp+instance_name + ".csv", orig_prob)
        #Tracking if instance autocompleted because bp=salbp and setting defaults
        cpu = -1 
        no_stations = salbp_sol

        #proceeds to precedence constraint removal, if bin_lb != no stations
        for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
            SALBP_dict = deepcopy(SALBP_dict_orig)
            SALBP_dict = precedence_removal(SALBP_dict, j)
            if bin_lb != salbp_sol: #If bin_lb==salbp_sol, then we don't need to do any precedence removal
                write_to_alb(SALBP_dict, temp_alb_path)
                output = subprocess.run([ex_fp, "-m", f"{branch}", "-b", "1", "-t", f"{time_limit}", temp_alb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            save_backup(out_fp+instance_name + ".csv", result)
            results.append(result)

    return results




def ils_call(cycle_time, tasks_times_list, precedence_list, 
                       max_iterations=1000, operation_probs=0.5, 
                       show_verbose=False, init_sol=None):
    """
    Uses ils to generate a SALBP solution
    
    Parameters:
    -----------
    cycle_time : int
        Maximum time allowed per workstation
    tasks_times_list : list of int
        Processing time for each task
    precedence_list : list of list of int
        Precedence constraints as [predecessor, successor] pairs
    max_iterations : int, optional
        Maximum iterations for the algorithm (default: 1000)
    operation_probs : float, optional
        Operation probabilities parameter (default: 0.5)
    show_verbose : bool, optional
        Whether to show verbose output (default: False)
    init_sol : list of int, optional
        Initial solution if available (default: None)
    
    Returns:
    --------
    ALBPSolution or None
        The solution object if successful, None if error occurs
    """
    
    if init_sol is None:
        init_sol = []
    
    N = len(tasks_times_list)
    
    try:
        solution = ils.ils_solve_SALBP1(
            C=cycle_time,
            N=N,
            task_times=tasks_times_list,
            raw_precedence=precedence_list,
            max_iter=max_iterations,
            op_probs=operation_probs,
            verbose=show_verbose,
            initial_solution=init_sol
        )
        
        if show_verbose:
            print(f"Successfully solved SALBP1 with {N} tasks and cycle time {cycle_time}")

        return solution

    except Exception as e:
        print(f"Error solving SALBP1: {e}")
        return None
    

def salbp1_mhh_solve(alb_dict,  
                **kwargs):
    #print(f"using alpha_iter {alpha_iter}, alpha_size {alpha_size}, beta_iter {beta_iter}, beta_size {beta_size}, reverse {reverse}")
    C = alb_dict['cycle_time']
    precs = alb_dict['precedence_relations']
    t_times = [val for _, val in alb_dict['task_times'].items()]
    N = len(t_times)
    precs = [[int(child), int(parent)]  for child, parent in alb_dict['precedence_relations']]
    start  = time.time()
    results = ils.mhh_solve_salbp1(
            C=C,
            N=N,
            task_times=t_times,
            raw_precedence=precs,

        )
    result_dict = results.to_dict()
    end = time.time()- start

    return {
            **result_dict,  "elapsed_time":end}
    

def salbp1_hoff_solve(alb_dict,  alpha_iter= 2,
                alpha_size = 0.05,
                beta_iter = None,
                beta_size = 0.005,
                reverse=True,
                **kwargs):
    #print(f"using alpha_iter {alpha_iter}, alpha_size {alpha_size}, beta_iter {beta_iter}, beta_size {beta_size}, reverse {reverse}")
    if not beta_iter:
        beta_iter = int(len(alb_dict['task_times'])/2)
    C = alb_dict['cycle_time']
    precs = alb_dict['precedence_relations']
    t_times = [val for _, val in alb_dict['task_times'].items()]
    N = len(t_times)
    precs = [[int(child), int(parent)]  for child, parent in alb_dict['precedence_relations']]
    start  = time.time()
    results = ils.hoff_solve_salbp1(
            C=C,
            N=N,
            task_times=t_times,
            raw_precedence=precs,
            alpha_iter = alpha_iter,
              alpha_size = alpha_size,
                beta_iter = beta_iter,
                beta_size = beta_size,
                reverse=reverse,

        )
    result_dict = results.to_dict()
    end = time.time()- start

    return {
            **result_dict,  "elapsed_time":end}

        
def salbp1_vdls_dict(alb_dict,time_limit=180, initial_solution = [], **mh_kwargs): 
    C = alb_dict['cycle_time']
    precs = alb_dict['precedence_relations']
    t_times = [val for _, val in alb_dict['task_times'].items()]
    N = len(t_times)
    precs = [[int(child), int(parent)]  for child, parent in alb_dict['precedence_relations']]
    start  = time.time()
    results = ils.vdls_solve_salbp1(C=C, N=N, task_times= t_times, raw_precedence=precs, max_attempts = 200000,time_limit = time_limit, initial_solution=initial_solution)
    result_dict = results.to_dict()
    end = time.time()- start

    return {
            **result_dict, "time_limit":time_limit, "elapsed_time":end}

       
def salbp1_prioirity_dict(alb_dict, n_random=100,**mh_kwargs):
    C = alb_dict['cycle_time']
    precs = alb_dict['precedence_relations']
    t_times = [val for _, val in alb_dict['task_times'].items()]
    N = len(t_times)
    precs = [[int(child), int(parent)]  for child, parent in alb_dict['precedence_relations']]
    start  = time.time()
    results = ils.priority_solve_salbp1(C=C, N=N, task_times= t_times, raw_precedence=precs, n_random=n_random )

    end = time.time()- start
    res_list = []
    for result in results:
        result_dict = {**result.to_dict(), "total_elapsed_time":end}
        res_list.append(result_dict)
    return res_list

def salbp1_prioirity_solve(alb_dict,time_limit=None, n_random=100, **kwargs):
    C = alb_dict['cycle_time']
    precs = alb_dict['precedence_relations']
    t_times = [val for _, val in alb_dict['task_times'].items()]
    N = len(t_times)
    precs = [[int(child), int(parent)]  for child, parent in alb_dict['precedence_relations']]
    start  = time.time()
    results = ils.priority_solve_salbp1(C=C, N=N, task_times= t_times, raw_precedence=precs, n_random=n_random )


    best = results[0]
    for result in results:
        if result.n_stations < best.n_stations:
            best=result
    
    end = time.time()- start
    best = {**best.to_dict(),"elapsed_time":end, "total_elapsed_time":end}
    return best


def mh_solve_edges(alb_dict, out_fp,mh_func,  time_limit, mh_config, **kwargs):
    xp_config, ml_config, ml_model = load_and_backup_configs(mh_config, backup_folder=out_fp)
    if mh_func == "salbp1_vdls_dict":
        mh_func= salbp1_vdls_dict
        mh_kwargs = xp_config['vdls']
    elif mh_func == "salbp1_priority_dict":
        mh_func = salbp1_prioirity_solve
        mh_kwargs = xp_config['priority']

    elif mh_func == "salbp1_hoff":
        mh_func = salbp1_hoff_solve
        mh_kwargs = xp_config['hoff']

        
    else:
        print(f"Error: given metaheuristic {mh_func} not supported")
    
    orig_prec = len(alb_dict['precedence_relations'])
    instance_fp = alb_dict['name']
    results = []
    # Extract instance name from file path
    instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]


    print("running: ", instance_name, " saving to output ", out_fp)
    # Use a unique temporary ALB file per process
    

    orig_solution = mh_func(
        alb_dict,
        time_limit,
        **mh_kwargs
    )
        
  
    orig_prob = {
        "instance": instance_name,
        "precedence_relation": "None",
        "nodes": "SALBP_original",
        "no_stations": orig_solution['n_stations'],
        "original_n_precedence_constraints": orig_prec,
      #  "optimal": "na",
        "cpu": orig_solution['elapsed_time'],
        "task_assignments": orig_solution['task_assignment']
        #"bin_lb": bin_lb
    }
    results.append(orig_prob)
    save_backup(out_fp+instance_name + ".csv", orig_prob)
    #Tracking if instance autocompleted because bp=salbp and setting defaults
    cpu = -1 
    no_stations = orig_solution['n_stations']

    #proceeds to precedence constraint removal, if bin_lb != no stations
    for j, relation in enumerate(alb_dict["precedence_relations"]):
        print("removing edge: ", relation)
        SALBP_dict = deepcopy(alb_dict)
        SALBP_dict = precedence_removal(SALBP_dict, j)
        solution =    mh_func(
        SALBP_dict,
        time_limit, 
        initial_solution=orig_solution['task_assignment'],
        **mh_kwargs
    )
        
        result = {
            "instance": instance_name,
            "precedence_relation": j,
            "nodes": relation,
            "no_stations": solution['n_stations'],
            "original_n_precedence_constraints": orig_prec,
            "cpu": solution['elapsed_time'],
            "task_assignments": solution['task_assignment']
        }
        save_backup(out_fp+instance_name + ".csv", result)
        results.append(result)

    return results

def generate_results_from_dict_list_2(alb_files, out_fp,  pool_size, mh_func, time_limit, mh_config):
    if not os.path.exists(out_fp):
         os.makedirs(out_fp)
    with multiprocessing.Pool(pool_size) as pool:
        results = pool.starmap(mh_solve_edges, [(alb, out_fp, mh_func, time_limit, mh_config) for alb in alb_files])
    #save_backup(out_fp + backup_name, results)
    return results

def generate_results_from_pickle_2(fp  ,out_fp, res_df ,    pool_size, start, stop , mh_func, time_limit, mh_config):
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
        instances = set(res_df['instance'])
        filtered_files = []
        for alb in alb_files:
            name = str(alb['name']).split('/')[-1].split('.')[0]
            if name not in instances:
                filtered_files.append(alb)
            else:
                print( name, "is already in results, skipping")

        results = generate_results_from_dict_list_2(filtered_files, out_fp, pool_size,mh_func, time_limit, mh_config)
    else:
        alb_files = open_salbp_pickle(fp)
        if start is not None and stop is not None:
            alb_files = alb_files[start:stop]
        results =  generate_results_from_dict_list_2(alb_files, out_fp,pool_size, mh_func, time_limit, mh_config)
    return results

def generate_results_from_dict_list(alb_files, out_fp, ex_fp, backup_name, pool_size, branch, time_limit):
    with multiprocessing.Pool(pool_size) as pool:
        results = pool.starmap(generate_one_instance_results, [(alb, ex_fp, out_fp, branch, time_limit) for alb in alb_files])

    save_backup(out_fp + backup_name, results)
    return results

def generate_results_from_pickle(fp  ,out_fp, res_df ,  ex_fp ,  backup_name , pool_size, start, stop , branch, time_limit):
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
        instances = set(res_df['instance'])
        filtered_files = []
        for alb in alb_files:
            name = str(alb['name']).split('/')[-1].split('.')[0]
            if name not in instances:
                filtered_files.append(alb)
            else:
                print( name, "is already in results, skipping")

        results = generate_results_from_dict_list(filtered_files, out_fp, ex_fp, backup_name, pool_size, branch=branch, time_limit = time_limit)
    else:
        alb_files = open_salbp_pickle(fp)
        if start is not None and stop is not None:
            alb_files = alb_files[start:stop]
        results =  generate_results_from_dict_list(alb_files, out_fp, ex_fp, backup_name, pool_size, branch=branch ,time_limit = time_limit)
    return results


def load_and_backup_configs(xp_config_fp, backup_folder="backups"):
    """
    Load experiment configuration, associated ML feature and model configs,
    and back up the main config file with today's date appended.

    Args:
        xp_config_fp (str or Path): Path to the main experiment config YAML.
        backup_folder (str or Path): Folder where backups will be stored.

    Returns:
        tuple: (xp_config, ml_config, ml_model)
    """
    xp_config_fp = Path(xp_config_fp)

    # --- Load experiment config ---
    with open(xp_config_fp, 'r') as file:
        xp_config = yaml.safe_load(file)
    print(f"Loaded experiment config from {xp_config_fp}")
    print(xp_config)

    # --- Load ML feature config ---
    ml_features_fp = Path(xp_config['ml_param_fp'])
    print(f"Using ML features and model specified in {ml_features_fp}")

    with open(ml_features_fp, 'r') as file:
        ml_config = yaml.safe_load(file)
    print(f"Loaded ML config from {ml_features_fp}")
    print(ml_config)

    # --- Load ML model ---
    ml_model_fp = Path(ml_config['ml_model_fp'])
    with open(ml_model_fp, 'rb') as f:
        ml_model = pickle.load(f)
    print(f"Loaded ML model from {ml_model_fp}")

    # --- Backup main config ---
    backup_folder = Path(backup_folder)
    backup_folder.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()  # e.g. "2025-10-24"
    backup_name = f"{xp_config_fp.stem}_{today}{xp_config_fp.suffix}"
    backup_path = backup_folder / backup_name

    shutil.copy2(xp_config_fp, backup_path)
    print(f"Backed up {xp_config_fp} → {backup_path}")

    return xp_config, ml_config, ml_model

# def generate_results_from_pickle(fp  ,out_fp,  ex_fp = "../BBR-for-SALBP1/SALB/SALB/salb",  backup_name = f"SALBP_edge_solutions.csv", pool_size = 4, start=None, stop=None):
#     results = []
#     #loads the pickle file
#     alb_files = open_salbp_pickle(fp)
#     if start is not None and stop is not None:
#         alb_files = alb_files[start:stop]


#     results = generate_results_from_dict_list(alb_files, out_fp, ex_fp, backup_name, pool_size)
#     return results

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Solve edge removal on SALBP instance')
    
    # Add arguments
    parser.add_argument('--start', type=int, required=False, help='Starting integer (inclusive)')
    parser.add_argument('--end', type=int, required=False, help='Ending integer (inclusive)')
    parser.add_argument('--n_processes', type=int, required=False, default=1, help='Number of processes to use')
    parser.add_argument('--from_alb_folder', action="store_true", help='Whether to read albs directly from a folder, if false, reads from pickle')
    parser.add_argument('--SALBP_solver_fp', type=str, default="../BBR-for-SALBP1/SALB/SALB/salb", help='Filepath for SALBP solver')
    parser.add_argument('--backup_name', type=str, required=False, default="results", help='name for intermediate saves')
    parser.add_argument('--mh_config_fp', type=str, required=False, default="results", help='filepath for mh config file')
    parser.add_argument('--filepath', type=str, required=True, help='filepath for alb dataset')
    parser.add_argument('--instance_name', type=str, required=False, help='start of instance name EX: "instance_n=50_"')
    parser.add_argument('--final_results_fp', type=str, required=True, help='filepath for results, if no error')
    parser.add_argument('--res_fp', type=str, required=False, help='Existing results df fp. Passing this filters out instances that have already been ran' )
    parser.add_argument('--solver_config', type=int, required=False, default=1, help='type of search strategy to use, 1 or 2 for the solver')
    parser.add_argument('--time_limit', type=int, required=False, default=1000, help='max time to solve the problem')
    parser.add_argument('--heuristic', type=str, required=False, help='what heuristic to use vdls or priority, none defaults to bbr')
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input
    if args.from_alb_folder:
        if args.instance_name is None:
            print("Error: Must provide instance name if not reading from pickle", file=sys.stderr)
            sys.exit(1)
        if args.start is None or args.end is None:
            print("Error: Must provide both start and end values if not reading from pickle", file=sys.stderr)
            sys.exit(1)
        if args.start > args.end:
            print("Error: Start value must be less than or equal to end value", file=sys.stderr)
            sys.exit(1)
        results = generate_results(fp = args.filepath, instance_name = args.instance_name, start=args.start, stop = args.end, backup_name=args.backup_name)

    else:
        if args.heuristic:
            if args.heuristic=='vdls':
                results = generate_results_from_pickle_2(args.filepath, args.final_results_fp,res_df = args.res_fp,  pool_size=args.n_processes, start=args.start, stop=args.end, mh_func= "salbp1_vdls_dict" , time_limit = args.time_limit, mh_config = args.mh_config_fp)
            elif args.heuristic=='priority':
                results = generate_results_from_pickle_2(args.filepath, args.final_results_fp,res_df = args.res_fp,  pool_size=args.n_processes, start=args.start, stop=args.end, mh_func= "salbp1_priority_dict" , time_limit = args.time_limit,  mh_config = args.mh_config_fp)
            elif args.heuristic=='hoff':
                results = generate_results_from_pickle_2(args.filepath, args.final_results_fp,res_df = args.res_fp,  pool_size=args.n_processes, start=args.start, stop=args.end, mh_func= "salbp1_hoff" , time_limit = args.time_limit,  mh_config = args.mh_config_fp)

        else:
            print("using bbr")
            results = generate_results_from_pickle(args.filepath, args.final_results_fp,res_df = args.res_fp, ex_fp=args.SALBP_solver_fp, backup_name=args.backup_name, pool_size=args.n_processes, start=args.start, stop=args.end, branch = args.solver_config, time_limit = args.time_limit)
    # Process the range
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.final_results_name)



if __name__ == "__main__":
    main()
#reads the results csv
#results_df = pd.read_csv("task_20_bin_lb.csv")
#results_df = pd.DataFrame(results)
#saves the results df to a csv file
#results_df.to_csv("tasks20_test.csv")
