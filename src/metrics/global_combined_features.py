# from node_and_edge_features import random_walk_from_node, generate_walk_stats_ALB
import networkx as nx
import pandas as pd
from SALBP_solve import salbp1_prioirity_dict
from salbp2_solve import salbp2_prioirity_dict
import time
from alb_instance_compressor import open_salbp_pickle
import math
from collections import defaultdict
import sys


def get_station_assignment_stats(priority_sols, cycle_time):
    """

    Computes mean/max/min/std for priority solutions
    """

    # Stats for each task
    # Each entry: {"count":..., "sum":..., "sum_sq":..., "max":..., "min":...}
    stats = {}

    for sol in priority_sols:
        station_loads = sol["loads"]
        station_assignments = sol["station_assignments"]

        for idx, assignment in enumerate(station_assignments):
            load_ratio = station_loads[idx] / cycle_time

            for task in assignment:
                if task not in stats:
                    stats[task] = {
                        "count": 0,
                        "sum": 0.0,
                        "sum_sq": 0.0,
                        "max": -float('inf'),
                        "min": float('inf')
                    }

                s = stats[task]
                s["count"] += 1
                s["sum"] += load_ratio
                s["sum_sq"] += load_ratio * load_ratio
                s["max"] = max(s["max"], load_ratio)
                s["min"] = min(s["min"], load_ratio)

    # Convert to desired output format
    result = {}
    for task, s in stats.items():
        c = s["count"]
        mean = s["sum"] / c
        # Var = E[x^2] - mean^2
        variance = (s["sum_sq"] / c) - (mean * mean)
        std = math.sqrt(variance) if variance > 0 else 0.0

        result[task] = {
            "mean": mean,
            "max": s["max"],
            "min": s["min"],
            "std": std
        }

    # Sorted keys like the old version
    result = dict(sorted(result.items(), key=lambda kv: kv[0]))

    return {"load_stats": result}


def generate_priority_sol_stats_salbp1(alb, n_random=100, generate_task_load_stats=True):
    total_start = time.time()
    timings = {}

    # ---------------- Priority solves ----------------
    t0 = time.time()
    priority_sols = salbp1_prioirity_dict(alb, n_random)
    timings["priority_solve"] = time.time() - t0

    # ---------------- Preprocessing ----------------
    t0 = time.time()
    task_times = [val for val in alb['task_times'].values()]
    t_div_c = sum(task_times) / alb['cycle_time']
    timings["preprocessing"] = time.time() - t0

    # ---------------- DataFrame creation ----------------
    t0 = time.time()
    priority_df = pd.DataFrame(priority_sols)
    timings["dataframe_creation"] = time.time() - t0

    # ---------------- Basic metrics ----------------
    t0 = time.time()
    min_sol = priority_df['n_stations'].min() 
    max_sol = priority_df['n_stations'].max()
    min_sol_gap = (min_sol - t_div_c)/t_div_c
    max_sol_gap = (max_sol - t_div_c)/t_div_c
    timings["basic_metrics"] = time.time() - t0

    # ---------------- Random stats ----------------
    t0 = time.time()
    random_df = priority_df[priority_df['method'].str.contains('random', case=False, na=False)]
    random_spread = random_df['n_stations'].max() - random_df['n_stations'].min()
    random_cv = random_df['n_stations'].std() / random_df['n_stations'].mean()
    random_avg_gap = (random_df['n_stations'].mean() - t_div_c) / t_div_c
    random_min_gap = (random_df['n_stations'].min() - t_div_c) / t_div_c
    random_max_gap = (random_df['n_stations'].max() - t_div_c) / t_div_c
    random_avg_efficiency = 1 - (
        (random_df['n_stations'].mean() * alb['cycle_time'] - sum(task_times)) /
        (random_df['n_stations'].mean() * alb['cycle_time'])
    )
    timings["random_metrics"] = time.time() - t0

    # ---------------- Build metrics dict ----------------
    t0 = time.time()
    metrics = {
        'priority_min_stations': min_sol,
        'priority_max_stations': max_sol,
        'priority_min_gap': min_sol_gap,
        'priority_max_gap': max_sol_gap,
        'random_spread': random_spread,
        'random_coefficient_of_variation': random_cv,
        'random_avg_gap': random_avg_gap,
        'random_min_gap': random_min_gap,
        'random_max_gap': random_max_gap,
        'random_avg_efficiency': random_avg_efficiency,
        'priority_calc_time': time.time() - total_start
    }
    timings["metrics_dict_build"] = time.time() - t0

    # ---------------- Compute task load stats ----------------
    if generate_task_load_stats:
        t0 = time.time()
        task_load_stats = get_station_assignment_stats(priority_sols, cycle_time=alb["cycle_time"]) 
        timings["task_load_stats"] = time.time() - t0

    # ---------------- Final timing report ----------------
    total_time = time.time() - total_start

    # print("\n=== Priority Stats Profiling ===")
    # for k, v in timings.items():
    #     pct = (v / total_time) * 100
    #     print(f"{k:25s}: {v:8.4f} sec   ({pct:5.1f}%)")
    # print(f"{'TOTAL':25s}: {total_time:8.4f} sec")
    # print("================================\n")

    # ---------------- Return ----------------
    if generate_task_load_stats:
        return {**metrics, **task_load_stats}
    return metrics

# def generate_priority_sol_stats_salbp1(alb, n_random=100, generate_task_load_stats=True):
#     start = time.time()
#     priority_sols = salbp1_prioirity_dict(alb, n_random)

#     task_times = [val for val in alb['task_times'].values()]
#     t_div_c = sum(task_times) / alb['cycle_time']
#     priority_df = pd.DataFrame(priority_sols)
#     min_sol = priority_df['n_stations'].min() 
#     max_sol = priority_df['n_stations'].max()
#     min_sol_gap = (min_sol - t_div_c)/t_div_c
#     max_sol_gap = (max_sol - t_div_c)/t_div_c
#     random_df = priority_df[priority_df['method'].str.contains('random', case=False, na=False)]
#     random_spread = ( random_df['n_stations'].max()-random_df['n_stations'].min())
#     random_cv = random_df['n_stations'].std()/random_df['n_stations'].mean()
#     random_avg_gap = (random_df['n_stations'].mean()-t_div_c)/t_div_c
#     random_min_gap = (random_df['n_stations'].min()-t_div_c)/t_div_c
#     random_max_gap = (random_df['n_stations'].max()-t_div_c)/t_div_c
#     random_avg_efficiency = 1- (random_df['n_stations'].mean()*alb['cycle_time'] - sum(task_times))/(random_df['n_stations'].mean()*alb['cycle_time'])
#     metrics= {
#     'priority_min_stations': min_sol,
#     'priority_max_stations': max_sol,
#     'priority_min_gap': min_sol_gap,
#     'priority_max_gap': max_sol_gap,
#     'random_spread': random_spread,
#     'random_coefficient_of_variation': random_cv,
#     'random_avg_gap': random_avg_gap,
#     'random_min_gap': random_min_gap,
#     'random_max_gap': random_max_gap,
#     'random_avg_efficiency': random_avg_efficiency,
#     'priority_calc_time': time.time()-start
#         }
#     if generate_task_load_stats:
#         task_load_stats = get_station_assignment_stats(priority_sols, cycle_time=alb["cycle_time"])
#         return {**metrics, **task_load_stats}
#     return metrics



def calc_global_combined_features_salbp1(alb_instance):
   priority_metrics = generate_priority_sol_stats_salbp1(alb_instance)

   return priority_metrics



def generate_priority_sol_stats_salbp2(alb, n_random=100):
    start = time.time()
    priority_sols = salbp2_prioirity_dict(alb, n_random)
    task_times = [val for val in alb['task_times'].values()]
    t_div_s = sum(task_times) / alb['n_stations']
    priority_df = pd.DataFrame(priority_sols)
    min_sol = priority_df['cycle_time'].min() 
    max_sol = priority_df['cycle_time'].max()
    min_sol_gap = (min_sol - t_div_s)/t_div_s
    max_sol_gap = (max_sol - t_div_s)/t_div_s
    random_df = priority_df[priority_df['method'].str.contains('random', case=False, na=False)]
    random_spread = ( random_df['cycle_time'].max()-random_df['cycle_time'].min())
    random_cv = random_df['cycle_time'].std()/random_df['cycle_time'].mean()
    random_avg_gap = (random_df['cycle_time'].mean()-t_div_s)/t_div_s
    random_min_gap = (random_df['cycle_time'].min()-t_div_s)/t_div_s
    random_max_gap = (random_df['cycle_time'].max()-t_div_s)/t_div_s
    random_avg_efficiency = 1- (random_df['cycle_time'].mean()*alb['n_stations'] - task_times)/(random_df['cycle_time'].mean()*alb['n_stations'])
    metrics= {
    'priority_min_c': min_sol,
    'priority_max_c': max_sol,
    'priority_min_gap': min_sol_gap,
    'priority_max_gap': max_sol_gap,
    'random_spread': random_spread,
    'random_coefficient_of_variation': random_cv,
    'random_avg_gap': random_avg_gap,
    'random_min_gap': random_min_gap,
    'random_max_gap': random_max_gap,
    'random_avg_efficiency': random_avg_efficiency,
    'priority_calc_time': time.time()-start
        }
    return metrics

def calc_global_combined_features_salbp2(orig_instance, S=None, n_random=100):
    alb_instance = orig_instance.copy()
    if not S:
        if not alb_instance.haskey('n_stations') :
            print("Error: must specify number of stations, alb_instance doesn't have it")
    else:
         alb_instance['n_stations'] = S
    priority_metrics = generate_priority_sol_stats_salbp2(alb_instance, n_random)
    return priority_metrics



