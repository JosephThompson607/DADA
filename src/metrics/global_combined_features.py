# from node_and_edge_features import random_walk_from_node, generate_walk_stats_ALB
import networkx as nx
import pandas as pd
from SALBP_solve import salbp1_prioirity_dict
from salbp2_solve import salbp2_prioirity_dict
import time
from alb_instance_compressor import open_salbp_pickle


def get_station_assignment_stats(priority_sols, cycle_time):
    """Generate features for edges based where they fell in solutions"""
    task_loads = []

    for sol in priority_sols:
        station_loads = sol["loads"]
        station_assignments = sol["station_assignments"]

        task_load_dict = {}
        for idx, station_assignment in enumerate(station_assignments):
            for task in station_assignment:
                task_load_dict[task] = station_loads[idx] / cycle_time
        task_loads.append(task_load_dict)
    # Build dataframe
    task_load_df = pd.DataFrame(task_loads)

    # Aggregate stats per task
    task_load_stats = task_load_df.aggregate(['mean', 'max', 'min','std'])

    # Transpose so tasks become the keys
    task_load_stats = task_load_stats.T
    task_load_stats.sort_index(ascending=True, inplace=True)

    stats_dict = task_load_stats.to_dict(orient="index")

    return {"load_stats":stats_dict}



def generate_priority_sol_stats_salbp1(alb, n_random=100, generate_task_load_stats=True):
    start = time.time()
    priority_sols = salbp1_prioirity_dict(alb, n_random)

    task_times = [val for val in alb['task_times'].values()]
    t_div_c = sum(task_times) / alb['cycle_time']
    priority_df = pd.DataFrame(priority_sols)
    min_sol = priority_df['n_stations'].min() 
    max_sol = priority_df['n_stations'].max()
    min_sol_gap = (min_sol - t_div_c)/t_div_c
    max_sol_gap = (max_sol - t_div_c)/t_div_c
    random_df = priority_df[priority_df['method'].str.contains('random', case=False, na=False)]
    random_spread = ( random_df['n_stations'].max()-random_df['n_stations'].min())
    random_cv = random_df['n_stations'].std()/random_df['n_stations'].mean()
    random_avg_gap = (random_df['n_stations'].mean()-t_div_c)/t_div_c
    random_min_gap = (random_df['n_stations'].min()-t_div_c)/t_div_c
    random_max_gap = (random_df['n_stations'].max()-t_div_c)/t_div_c
    random_avg_efficiency = 1- (random_df['n_stations'].mean()*alb['cycle_time'] - sum(task_times))/(random_df['n_stations'].mean()*alb['cycle_time'])
    metrics= {
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
    'priority_calc_time': time.time()-start
        }
    if generate_task_load_stats:
        task_load_stats = get_station_assignment_stats(priority_sols, cycle_time=alb["cycle_time"])
        return {**metrics, **task_load_stats}
    return metrics

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



