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
    sum_div_c = np.sum(task_times) / C
    std_div_c = np.std(task_times) / C
    return {'min_div_c': min_div_c, 'max_div_c': max_div_c, 'sum_div_c': sum_div_c, 'std_div_c': std_div_c}