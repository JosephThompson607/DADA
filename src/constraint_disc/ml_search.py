import pandas as pd
import sys

sys.path.append('src')
from data_prep import albp_to_features
import copy

def predictor(orig_salbp, G_max_red, ml_model, ml_feature_list=None):
    #Take task times form base salbp problem and use them with the new edges
    test_salbp = copy.deepcopy(orig_salbp)
    test_salbp['precedence_relations'] = list(G_max_red.edges())
    edge_features = albp_to_features(test_salbp, salbp_type="salbp_1", cap_constraint = None, n_random=100)
    edge_names = edge_features['edge']
    if ml_feature_list:
        features = ml_feature_list
    else:
        features =[
                                    'parent_weight', 'parent_pos_weight',
                                        'child_weight', 'child_pos_weight', 'neighborhood_min',
                                        'neighborhood_max', 'neighborhood_avg', 'neighborhood_std',
                                        'parent_in_degree', 'parent_out_degree', 'child_in_degree',
                                        'child_out_degree', 'chain_avg', 'chain_min', 'chain_max', 'chain_std',
                                        'edge_data_time', 'rw_mean_total_time', 'rw_mean_min_time',
                                        'rw_mean_max_time', 'rw_mean_n_unique_nodes',
                                        'rw_min', 'rw_max', 'rw_mean', 'rw_std', 'rw_n_unique_nodes', 'child_rw_mean_total_time', 'child_rw_mean_min_time',
                                        'child_rw_mean_max_time', 'child_rw_mean_n_unique_nodes', 'child_rw_min', 'child_rw_max',
                                        'child_rw_mean', 'child_rw_std', 'child_rw_n_unique_nodes', 'priority_min_gap', 'priority_max_gap',
                                        'random_spread', 'random_coefficient_of_variation', 'random_avg_gap',
                                        'random_min_gap', 'random_max_gap', 
                                            'ti_size',
                                            'prec_bias',
                                            'stage_difference',
                                            'prec_strength',
                                            'load_parent_mean' ,
                                            'load_parent_max' ,
                                            'load_parent_min' ,
                                            'load_parent_std' ,
                                            'load_child_mean' ,
                                            'load_child_max' ,
                                            'load_child_min' ,
                                            'load_child_std' ,
                                        'min_div_c', 'max_div_c',  'std_div_c','avg_div_c',
                                        'order_strength', 'average_number_of_immediate_predecessors',
                                        'max_degree', 'max_in_degree', 'max_out_degree', 'divergence_degree',
                                        'convergence_degree',  'share_of_bottlenecks',
                                            'avg_chain_length',
                                        'nodes_in_chains', 'stages_div_n', 'n_isolated_nodes',
                                        'share_of_isolated_nodes', 'n_tasks_without_predecessors',
                                        'share_of_tasks_without_predecessors', 'avg_tasks_per_stage',
                                    ] 
    X_all = edge_features[features]
    y_prob = ml_model.predict_proba(X_all)[:, 1]
    edge_prob_df = pd.DataFrame({
        'edge': edge_names,
        'predicted_probability': y_prob
    })
    return edge_prob_df


def select_best_edge(edge_prob_df, valid_edges):
    """
    Filter edge_prob_df to only include edges in valid_edges,
    then return the edge with the highest predicted probability.
    """
    # Filter DataFrame by valid edges
    valid_edges = set([(str(e1),str(e2)) for (e1,e2) in valid_edges])
    edge_prob_df['valid'] = edge_prob_df['edge'].apply(lambda x: (str(x[0]), str(x[1])) in valid_edges)
    filtered = edge_prob_df[edge_prob_df['valid']==True]
    if filtered.empty:
        print("Error: No edges in the probability dataframe ")
        return None  # or raise an exception if that’s unexpected

    # Select edge with max probability
    best_row = filtered.loc[filtered['predicted_probability'].idxmax()]
    return best_row['edge'], best_row['predicted_probability']


def best_first_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model,**new_kwargs):
    '''Selects the edge with the highest probability of reducing the objective value'''

    prob_df = predictor(orig_salbp, G_max_red, ml_model)
    best_edge = select_best_edge(prob_df, edges)

    return best_edge
