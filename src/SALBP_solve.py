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
from alb_instance_compressor import parse_alb, write_to_alb, open_salbp_pickle, get_instance_name
import subprocess
import pandas as pd
from copy import deepcopy
import ILS_ALBP as ils 
import time
from functools import partial

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



# def generate_one_instance_results(alb_dict, ex_fp):
#     results = []
#     SALBP_dict_orig = alb_dict
#     bin_dict = deepcopy(SALBP_dict_orig)
#     instance_fp = SALBP_dict_orig['name']
#     #instance name is after the last / and before the .alb
#     instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]
#     print("instance name", instance_fp)
#     for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
#         SALBP_dict = deepcopy(SALBP_dict_orig)
#         SALBP_dict =precedence_removal(SALBP_dict, j)
#         write_to_alb(SALBP_dict, "test.alb")
#         output = subprocess.run([ex_fp,"-m","2", "test.alb"], stdout=subprocess.PIPE)
#         no_stations, optimal, cpu = parse_bb_salb1_out(output)
#         result = {"instance:": f"{instance_name}", "precedence_relation": j, "nodes": relation,  "no_stations": no_stations, "optimal": optimal, "cpu": cpu}
#         save_backup(instance_name +".csv", result)
#         results.append(result)

#     #calculates bin packing lower bound
#     bin_dict['precedence_relations'] = []
#     write_to_alb(bin_dict, "test.alb")
#     #TODO ad -m 2 to hopefully avoid bug in solver code
#     output = subprocess.run([ex_fp, "-m","2", "test.alb"], stdout=subprocess.PIPE)
#     no_stations, optimal, cpu = parse_bb_salb1_out(output)
#     result = {"instance": f"{instance_name}", "precedence_relation": "None", "no_stations": no_stations, "optimal": optimal, "cpu": cpu}
#     save_backup( instance_name+".csv", result)
#     results.append(result)
#     return results

def generate_one_instance_results(alb_dict, ex_fp, out_fp, branch=1):
    SALBP_dict_orig = alb_dict
    bin_dict = deepcopy(SALBP_dict_orig)
    instance_fp = SALBP_dict_orig['name']
    results = []
    # Extract instance name from file path
    instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]

    if not os.path.exists(out_fp):
         os.makedirs(out_fp)
    print("running: ", instance_name, " saving to output ", out_fp)
    # Use a unique temporary ALB file per process
    with tempfile.NamedTemporaryFile(suffix=".alb", delete=True) as temp_alb:
        temp_alb_path = temp_alb.name  # Path to temporary file
        orig_prec = len(SALBP_dict_orig["precedence_relations"])
        #original problem
        SALBP_dict = deepcopy(SALBP_dict_orig)
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
        save_backup(out_fp+instance_name + ".csv", orig_prob)
        #Tracking if instance autocompleted because bp=salbp and setting defaults
        cpu = -1 
        no_stations = salbp_sol

        #proceeds to precedence constraint removal, if bin_lb != no stations
        for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
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
            save_backup(out_fp+instance_name + ".csv", result)
            results.append(result)

    return results

def use_ils_on_albs(instance_dicts, pool_size =4, max_iterations = 20000):
    
    
    with multiprocessing.Pool(pool_size) as pool:
        partial_func = partial(run_alb_ils_dict, max_iterations = max_iterations)
        results = pool.map(partial_func, instance_dicts)
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
        
        




def ils_bbr_solve_edges(alb_dict, ex_fp, out_fp, branch=1, method=ils_call):
    SALBP_dict_orig = alb_dict
    orig_prec = len(alb_dict['precedence_relations'])
    instance_fp = SALBP_dict_orig['name']
    results = []
    # Extract instance name from file path
    instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]

    if not os.path.exists(out_fp):
         os.makedirs(out_fp)
    print("running: ", instance_name, " saving to output ", out_fp)
    # Use a unique temporary ALB file per process
    
    start = time.time()
    solution = ils.ils_solve_SALBP1(
            C=C,
            N=N,
            task_times=task_times,
            raw_precedence=raw_precedence,
            max_iter=max_iter,
            op_probs=op_probs,
            verbose=verbose,
            initial_solution=initial_solution
        )
        
    end = time.time()
  
    orig_prob = {
        "instance": instance_name,
        "precedence_relation": "None",
        "nodes": "SALBP_original",
        "no_stations": solution.n_stations,
        "original_n_precedence_constraints": orig_prec,
      #  "optimal": "na",
        "cpu": time,
        #"bin_lb": bin_lb
    }
    results.append(orig_prob)
    save_backup(out_fp+instance_name + ".csv", orig_prob)
    #Tracking if instance autocompleted because bp=salbp and setting defaults
    cpu = -1 
    no_stations = salbp_sol

    #proceeds to precedence constraint removal, if bin_lb != no stations
    for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
        print("removing edge: ", relation)
        SALBP_dict = deepcopy(SALBP_dict_orig)
        SALBP_dict = precedence_removal(SALBP_dict, j)
        
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


def generate_results_from_dict_list(alb_files, out_fp, ex_fp="../BBR-for-SALBP1/SALB/SALB/salb", backup_name="SALBP_edge_solutions.csv", pool_size=4, branch=1):
    with multiprocessing.Pool(pool_size) as pool:
        results = pool.starmap(generate_one_instance_results, [(alb, ex_fp, out_fp, branch) for alb in alb_files])

    save_backup(out_fp + backup_name, results)
    return results

def generate_results_from_pickle(fp  ,out_fp, res_df = None,  ex_fp = "../BBR-for-SALBP1/SALB/SALB/salb",  backup_name = f"SALBP_edge_solutions.csv", pool_size = 4, start=None, stop=None , branch=1):
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

        results = generate_results_from_dict_list(filtered_files, out_fp, ex_fp, backup_name, pool_size, branch=branch)
    else:
        alb_files = open_salbp_pickle(fp)
        if start is not None and stop is not None:
            alb_files = alb_files[start:stop]
        results =  generate_results_from_dict_list(alb_files, out_fp, ex_fp, backup_name, pool_size, branch=branch)
    return results




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
    parser.add_argument('--backup_name', type=str, required=True, help='name for intermediate saves')
    parser.add_argument('--filepath', type=str, required=True, help='filepath for alb dataset')
    parser.add_argument('--instance_name', type=str, required=False, help='start of instance name EX: "instance_n=50_"')
    parser.add_argument('--final_results_fp', type=str, required=True, help='filepath for results, if no error')
    parser.add_argument('--res_fp', type=str, required=False, help='Existing results df fp. Passing this filters out instances that have already been ran' )
    parser.add_argument('--solver_config', type=int, required=False, default=1, help='type of search strategy to use, 1 or 2 for the solver')
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
        results = generate_results_from_pickle(args.filepath, args.final_results_fp,res_df = args.res_fp, ex_fp=args.SALBP_solver_fp, backup_name=args.backup_name, pool_size=args.n_processes, start=args.start, stop=args.end, branch = args.solver_config)
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
