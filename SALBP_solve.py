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



def compress_alb_instances(dir, output_fp):
    #gets all files with .alb extension, checks subdirectories
    files = [f for f in Path(dir).rglob("*.alb")]
    
    alb_files = []
    for f in files:
        alb_file = parse_alb(f"{f}")
        alb_file['name'] = f
        alb_files.append(alb_file)

    #saves to pickle
    with open(output_fp, 'wb') as f:
        pickle.dump(alb_files, f)

def parse_alb(alb_file_name):
    """Reads assembly line balancing instance .alb file, returns dictionary with the information"""
    parse_dict = {}
    alb_file = open(alb_file_name).read()
    # Get number of tasks
    num_tasks = re.search("<number of tasks>\n(\\d*)", alb_file)
    parse_dict["num_tasks"] = int(num_tasks.group(1))

    # Get cycle time
    cycle_time = re.search("<cycle time>\n(\\d*)", alb_file)
    parse_dict["cycle_time"] = int(cycle_time.group(1))

    # Order Strength
    # order_strength = re.search("<order strength>\n(\\d*,\\d*)", alb_file)

    # if order_strength is  not None:
    #     order_strength = re.search("<order strength>\n(\\d*.\\d*)", alb_file)
    #     parse_dict["original_order_strength"] = float(order_strength.group(1))

    # Task_times
    task_times = re.search("<task times>(.|\n)+?<", alb_file)

    # Get lines in this regex ignoring the first and last 2
    task_times = task_times.group(0).split("\n")[1:-2]
    task_times = {task.split()[0]: int(task.split()[1]) for task in task_times}
    parse_dict["task_times"] = task_times

    # Precedence relations
    precedence_relations = re.search("<precedence relations>(.|\n)+?<", alb_file)
    precedence_relations = precedence_relations.group(0).split("\n")[1:-2]
    precedence_relations = [task.split(",") for task in precedence_relations]
    parse_dict["precedence_relations"] = precedence_relations
    return parse_dict

#function that returns names of all files in a directory with a given extension
def get_instance_list(directory, keep_directory_location = True,  extension='.alb'):
    if keep_directory_location:
        return [ directory + '/' + f for f in os.listdir(directory) if f.endswith(extension)]
    else:
        return [f for f in os.listdir(directory) if f.endswith(extension)]

def write_to_alb(salbp_dict, alb_file_name):
    """Writes the SALBP dictionary to an .alb file"""
    #Format of alb:
    # <number of tasks>
    # no_tasks
    # <cycle time>
    # cycle_time
    #<task times>
    #task_id task_time
    #<precedence relations>
    #task_id,task_id
    #if salb_dict has key n_tasks, change it to num_tasks
    if "n_tasks" in salbp_dict:
        salbp_dict["num_tasks"] = salbp_dict["n_tasks"]
        salbp_dict.pop("n_tasks")
    if "num_tasks" not in salbp_dict:
        salbp_dict["num_tasks"] = len(salbp_dict["task_times"])

    # Write number of tasks
    alb = "<number of tasks>\n"
    alb += str(salbp_dict["num_tasks"]) + "\n"
    # Write cycle time
    alb += "<cycle time>\n"
    alb += str(salbp_dict["cycle_time"]) + "\n"
    # Write task times
    alb += "<task times>\n"
    for task_id, task_time in salbp_dict["task_times"].items():
        alb += task_id + " " + str(task_time) + "\n"
    # Write precedence relations
    alb += "<precedence relations>\n"
    for relation in salbp_dict["precedence_relations"]:
        alb += str(relation[0]) + "," + str(relation[1]) + "\n"
    #ends the file
    alb += "<end>"
    with open(alb_file_name, "w") as alb_file:
        alb_file.write(alb)
    


import subprocess
import pandas as pd
from copy import deepcopy

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
    print("precedence_relations", len(SALBP_dict["precedence_relations"]))
    print("edge_index", edge_index)
    SALBP_dict["precedence_relations"].pop(edge_index)
    return SALBP_dict
    

def parse_bb_salb1_out(text):
    '''gets the number of stations, optimal flag and cpu time from the output of the salb1 program'''
    output = text.stdout.decode("utf-8")
    # Regular expression to capture the required values
    match = re.search(r"verified_optimality\s*=\s*(\d+);\s*value\s*=\s*(\d+);\s*cpu\s*=\s*([\d.]+)", output)

    if match:
        verified_optimality = int(match.group(1))
        value = int(match.group(2))
        cpu = float(match.group(3))

        print("verified_optimality:", verified_optimality)
        print("value:", value)
        print("cpu:", cpu)
    else:
        print("Pattern not found.")
        print("output", output)
        value = -1000
        verified_optimality = 0
        cpu = -1000
    return value, verified_optimality, cpu





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
        intermediate.to_csv(backup_name, mode='a', header=False)
    else:
        intermediate.to_csv(backup_name)
    
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

def generate_one_instance_results(alb_dict, ex_fp):
    results = []
    SALBP_dict_orig = alb_dict
    bin_dict = deepcopy(SALBP_dict_orig)
    instance_fp = SALBP_dict_orig['name']
    
    # Extract instance name from file path
    instance_name = str(instance_fp).split("/")[-1].split(".alb")[0]

    # Use a unique temporary ALB file per process
    with tempfile.NamedTemporaryFile(suffix=".alb", delete=True) as temp_alb:
        temp_alb_path = temp_alb.name  # Path to temporary file

        for j, relation in enumerate(SALBP_dict_orig["precedence_relations"]):
            SALBP_dict = deepcopy(SALBP_dict_orig)
            SALBP_dict = precedence_removal(SALBP_dict, j)
            
            write_to_alb(SALBP_dict, temp_alb_path)
            output = subprocess.run([ex_fp, "-m", "2", temp_alb_path], stdout=subprocess.PIPE)
            
            no_stations, optimal, cpu = parse_bb_salb1_out(output)
            result = {
                "instance": instance_name,
                "precedence_relation": j,
                "nodes": relation,
                "no_stations": no_stations,
                "optimal": optimal,
                "cpu": cpu
            }
            save_backup(instance_name + ".csv", result)
            results.append(result)

        # Compute bin packing lower bound
        bin_dict['precedence_relations'] = []
        write_to_alb(bin_dict, temp_alb_path)
        output = subprocess.run([ex_fp, "-m", "2", temp_alb_path], stdout=subprocess.PIPE)
        
        no_stations, optimal, cpu = parse_bb_salb1_out(output)
        result = {
            "instance": instance_name,
            "precedence_relation": "None",
            "no_stations": no_stations,
            "optimal": optimal,
            "cpu": cpu
        }
        save_backup(instance_name + ".csv", result)
        results.append(result)

    return results

def generate_results_from_dict_list(alb_files, out_fp, ex_fp="../BBR-for-SALBP1/SALB/SALB/salb", backup_name="SALBP_edge_solutions.csv", pool_size=4):
    with multiprocessing.Pool(pool_size) as pool:
        results = pool.starmap(generate_one_instance_results, [(alb, ex_fp) for alb in alb_files])

    save_backup(out_fp + backup_name, results)
    return results


def generate_results_from_pickle(fp  ,out_fp,  ex_fp = "../BBR-for-SALBP1/SALB/SALB/salb",  backup_name = f"SALBP_edge_solutions.csv", pool_size = 4, start=None, stop=None):
    results = []
    #loads the pickle file
    with open(fp, 'rb') as f:
        alb_files = pickle.load(f)
    if start is not None and stop is not None:
        alb_files = alb_files[start:stop]


    results = generate_results_from_dict_list(alb_files, out_fp, ex_fp, backup_name, pool_size)
    return results

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Solve edge removal on SALBP instance')
    
    # Add arguments
    parser.add_argument('--start', type=int, required=False, help='Starting integer (inclusive)')
    parser.add_argument('--end', type=int, required=False, help='Ending integer (inclusive)')
    parser.add_argument('--n_processes', type=int, required=False, default=1, help='Number of processes to use')
    parser.add_argument('--from_alb_folder', action="store_true", help='Whether to read albs directly from a folder, if false, reads from pickle')
    parser.add_argument('--SALBP_solver_fp', type=str, default="BBR-for-SALBP1/SALB/SALB/salb", help='Filepath for SALBP solver')
    parser.add_argument('--backup_name', type=str, required=True, help='name for intermediate saves')
    parser.add_argument('--filepath', type=str, required=True, help='filepath for alb dataset')
    parser.add_argument('--instance_name', type=str, required=False, help='start of instance name EX: "instance_n=50_"')
    parser.add_argument('--final_results_fp', type=str, required=True, help='filepath for results, if no error')
    
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
        results = generate_results_from_pickle(args.filepath, args.final_results_fp,ex_fp=args.SALBP_solver_fp, backup_name=args.backup_name, pool_size=args.n_processes, start=args.start, stop=args.end)
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
