
import os
import pickle
from pathlib import Path
import glob
import sys
import re
import argparse
import pandas as pd

def open_salbp_pickle(fp):
    with open(fp, 'rb') as f:
        alb_files = pickle.load(f)

    #checks if the pickle file is a list of dictionaries
    if not isinstance(alb_files, list) and all(isinstance(x, dict) for x in alb_files):
        print("Error: Pickle file does not contain a list of dictionaries", file=sys.stderr)
        return None
    #checks and makes sure the first dictionary has the keys task_times, precedence_relations, cycle_time
    if not all(key in alb_files[0] for key in ["task_times", "precedence_relations", "cycle_time"]):
        print("Error: First dictionary in pickle file does not have the keys task_times, precedence_relations, cycle_time", file=sys.stderr)
    return alb_files

def create_pickle_df(fp):
    alb_dicts = open_salbp_pickle(fp)
    instance_names = [str(alb['name']).split('/')[-1].split('.')[0] for alb in alb_dicts]
    for alb, name in zip(alb_dicts, instance_names):
        alb['instance_name'] = name
    # Convert the updated list of dicts into a DataFrame
    pkl_df = pd.DataFrame(alb_dicts)
    pkl_df['precedence_relation'] = pkl_df['precedence_relations'].apply(enumerate_list)
    pkl_df= pkl_df.explode('precedence_relation')
    pkl_df['edge_idx'] = pkl_df['precedence_relation'].apply(lambda x: x[-1])
    pkl_df['precedence_relation'] = pkl_df['precedence_relation'].apply(lambda x: x[:2])
    return pkl_df

def get_instance_name(salbp_dict):
    '''Processes pickled instnaces'''
    instance_name = str(salbp_dict['name']).split('/')[-1].split('.')[0]
    return instance_name


def parse_fp_to_name(fp):
    '''Processes pickled instnaces'''
    instance_name = str(fp).split('/')[-1].split('.')[0]
    return instance_name

def open_salbp_pickle_as_dict(fp):
    with open(fp, 'rb') as f:
        alb_files = pickle.load(f)
    
    #checks if the pickle file is a list of dictionaries
    if not isinstance(alb_files, list) and all(isinstance(x, dict) for x in alb_files):
        print("Error: Pickle file does not contain a list of dictionaries", file=sys.stderr)
        return None
    #checks and makes sure the first dictionary has the keys task_times, precedence_relations, cycle_time
    if not all(key in alb_files[0] for key in ["task_times", "precedence_relations", "cycle_time", "name"]):
        print("Error: First dictionary in pickle file does not have the keys task_times, precedence_relations, cycle_time, or name", file=sys.stderr)
    salbp_dict = {get_instance_name(inst):inst for inst in alb_files}
    return salbp_dict

def open_multi_pickles_as_dict(pickles_fp_list):
    all_instances = {}
    for fp in pickles_fp_list:
        pickle_name = fp.split('/')[-1].split('.')[0]
        all_instances[pickle_name]=open_salbp_pickle_as_dict(fp)
    return all_instances

import re
import sys

def parse_alb(alb_file_name):
    """Reads assembly line balancing instance .alb file, returns dictionary with the information"""
    parse_dict = {}
    
    with open(alb_file_name, "r") as f:
        alb_file = f.read()

    # Number of tasks
    num_tasks = re.search(r"<number of tasks>\s*(\d+)", alb_file, re.IGNORECASE)
    if not num_tasks:
        print(f"Error: Number of tasks not found in {alb_file_name}", file=sys.stderr)
        return None
    parse_dict["num_tasks"] = int(num_tasks.group(1))

    # Cycle time
    cycle_time = re.search(r"<cycle time>\s*(\d+)", alb_file, re.IGNORECASE)
    if not cycle_time:
        print(f"Error: Cycle time not found in {alb_file_name}", file=sys.stderr)
        return None
    parse_dict["cycle_time"] = int(cycle_time.group(1))

    # Task times
    task_times_block = re.search(r"<task times>(.+?)<", alb_file, re.DOTALL | re.IGNORECASE)
    if not task_times_block:
        print(f"Error: Task times not found in {alb_file_name}", file=sys.stderr)
        return None
    task_lines = task_times_block.group(1).strip().splitlines()
    task_times = {}
    for line in task_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            task_times[parts[0]] = int(parts[1])
    parse_dict["task_times"] = task_times

    # Precedence relations
    precedence_block = re.search(r"<precedence relations>(.+?)<", alb_file, re.DOTALL | re.IGNORECASE)
    if not precedence_block:
        print(f"Error: Precedence relations not found in {alb_file_name}", file=sys.stderr)
        return None
    precedence_lines = precedence_block.group(1).strip().splitlines()
    precedence_relations = []
    for line in precedence_lines:
        parts = [p.strip() for p in line.strip().split(",") if p.strip()]
        if parts:
            precedence_relations.append(parts)
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

def get_folders_with_alb_files(root_path):
    folders = set()
    for filepath in glob.iglob(os.path.join(root_path, '**', '*.alb'), recursive=True):
        folders.add(os.path.dirname(filepath))
    return list(folders)

def find_and_compress_alb_instances(dir, output_fp, overwrite=False):
    #gets all folders with .alb extension
    alb_files = get_folders_with_alb_files(dir)
    for folder in alb_files:
        print(f"Compressing {folder}")
        #checks if the output file already exists
        if os.path.exists(output_fp+folder.split("/")[-1]+".pkl"):
            print(f"Error: {output_fp+folder.split('/')[-1]+'.pkl'} already exists", file=sys.stderr)
            if not overwrite:
                print("Overwrite flag not set, skipping", file=sys.stderr)
                continue
        compress_alb_instances(folder, output_fp+folder.split("/")[-1]+".pkl")


def compress_alb_instances(dir, output_fp):
    #gets all files with .alb extension, checks subdirectories
    files = [f for f in Path(dir).rglob("*.alb")]
    
    alb_files = []
    for f in files:
        alb_file = parse_alb(f"{f}")
        if alb_file is None:
            continue
        alb_file['name'] = f
        alb_files.append(alb_file)

    #saves to pickle
    with open(output_fp, 'wb') as f:
        pickle.dump(alb_files, f)


def main():
    #reads command line arguments
    parser = argparse.ArgumentParser(description='Compresses assembly line balancing instances')
    parser.add_argument('input', type=str, help='Directory containing .alb files')
    parser.add_argument('output', type=str, help='Output file path')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    args = parser.parse_args()
    #checks if the input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Directory {args.input} does not exist", file=sys.stderr)
        return
    #checks if the output directory exists
    if not os.path.exists(os.path.dirname(args.output)):
        print(f"Error: Directory {os.path.dirname(args.output)} does not exist", file=sys.stderr)
        return
    find_and_compress_alb_instances(args.input, args.output, args.overwrite)
    print("Done")



if __name__ == "__main__":
    main()