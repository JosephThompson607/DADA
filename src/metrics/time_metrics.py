import numpy as np



def get_time_stats(alb_instance, C=None):
    """
    Get time statistics for the given ALB instance.
    :param alb_instance: ALB instance
    :return: dictionary containing time statistics
    """
    if C is None:
        #if alb_instance['cycle_time'] is  not provide, raise an error
        if 'cycle_time' not in alb_instance:
            raise ValueError('Cycle time not provided, add value to C or to alb_instance')
        C = alb_instance['cycle_time']
    task_times = list( alb_instance['task_times'].values())
    min_div_c = np.min(task_times) / C
    max_div_c = np.max(task_times) / C
    time_interval_size = max_div_c - min_div_c
    sum_div_c = np.sum(task_times) / C
    std_div_c = np.std(task_times) / C
    avg_div_c = np.mean(task_times)/ C
    t_cv = np.std(task_times)/(sum(task_times)/len(task_times))
    return {'min_div_c': min_div_c, 'max_div_c': max_div_c, 'sum_div_c': sum_div_c, 'std_div_c': std_div_c, 't_cv':t_cv, 'ti_size':time_interval_size, 'avg_div_c':avg_div_c}



def get_time_stats_salb2(alb_instance, S=None):
    """
    Get time statistics for the given ALB instance.
    :param alb_instance: ALB instance
    :return: dictionary containing time statistics
    """
    if S is None:
        #if alb_instance['cycle_time'] is  not provide, raise an error
        if 'n_stations' not in alb_instance:
            raise ValueError('stations  not provided, add value to C or to alb_instance')
        S = alb_instance['n_stations']
    task_times = list( alb_instance['task_times'].values())
    min_div_s = np.min(task_times) / S
    max_div_s= np.max(task_times) / S
    sum_div_s = np.sum(task_times) / S
    std_div_s = np.std(task_times) / S
    t_cv = np.std(task_times)/(sum(task_times)/len(task_times))
    station_time_interval_size = max_div_s - min_div_s
    return {'n_stations':S,'min_div_s': min_div_s, 'max_div_s': max_div_s, 'sum_div_s': sum_div_s, 'std_div_s': std_div_s,'tis_size':station_time_interval_size, 't_cv':t_cv,'tasks_per_station':len(task_times)/S}