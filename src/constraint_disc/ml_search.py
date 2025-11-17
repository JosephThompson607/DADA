import pandas as pd
import sys

sys.path.append('src')
from data_prep import albp_to_features
import copy

def predictor(orig_salbp, G_max_red, ml_model, ml_config,**_):
    #Take task times form base salbp problem and use them with the new edges
    if 'n_edge_random' not in ml_config.keys():
        ml_config['n_edge_random'] = 0
    ml_feature_list = ml_config["features"]
    test_salbp = copy.deepcopy(orig_salbp)
    test_salbp['precedence_relations'] = list(G_max_red.edges())
    edge_features = albp_to_features(test_salbp, salbp_type="salbp_1", cap_constraint = None, n_random=ml_config['n_random'], feature_types=set(ml_config['feature_types']), n_edge_random=ml_config['n_edge_random'])
    edge_names = edge_features['edge']
    edge_features 
    features = ml_feature_list

    X_all = edge_features[features]
    y_prob = ml_model.predict_proba(X_all)[:, 1]
    edge_prob_df = pd.DataFrame({
        'edge': edge_names,
        'pred_val_prob': y_prob
    })
        # --- Add the existing 'prob' attribute from directed graph ---
    def get_edge_attributes(edge):
    # edge is assumed to be a tuple (u, v)
        if edge in G_max_red.edges:
            prob = G_max_red.edges[edge].get('prob', 1)
            t_cost = G_max_red.edges[edge].get('t_cost', 1)  # or whatever default you want
            return prob, t_cost
        else:
            return 1, 1  # default values
    edge_prob_df[['precedent_prob', 't_cost']] = edge_prob_df['edge'].apply(get_edge_attributes).apply(pd.Series)
    return edge_prob_df


# def select_best_edge(edge_prob_df, valid_edges):
#     """
#     Filter edge_prob_df to only include edges in valid_edges,
#     then return the edge with the highest predicted probability.
#     """
#     # Filter DataFrame by valid edges
#     valid_edges = set([(str(e[0]),str(e[1])) for e in valid_edges])
#     edge_prob_df['valid'] = edge_prob_df['edge'].apply(lambda x: (str(x[0]), str(x[1])) in valid_edges)
#     filtered = edge_prob_df[edge_prob_df['valid']==True]
#     if filtered.empty:
#         print("Error: No edges in the probability dataframe ")
#         return None  # or raise an exception if that’s unexpected

#     # Select edge with max probability
#     best_row = filtered.loc[filtered['pred_val_prob'].idxmax()]
#     return best_row['edge'], best_row['pred_val_prob']

def filter_for_valid_edges(valid_edges, edge_prob_df):
    valid_edges = set([(str(e[0]),str(e[1])) for e in valid_edges])
    edge_prob_df['valid'] = edge_prob_df['edge'].apply(lambda x: (str(x[0]), str(x[1])) in valid_edges)
    filtered = edge_prob_df[edge_prob_df['valid']==True]
    return filtered


def select_best_n_edges(edge_prob_df, valid_edges, top_n):
    """
    If given an edge probability feature, we multiply the ml value by the probability of the edge not existing. We then Filter edge_prob_df to only include edges in valid_edges,
    then return the edge with the highest predicted probability.

    Returns list of (edge, reward, and probability of edge existing) notice that the reward is prob of edge impacting solution * prob of edge existing
    """
    
    edge_prob_df['reward'] =  edge_prob_df['precedent_prob']* edge_prob_df['pred_val_prob']

    # Filter DataFrame by valid edges
    filtered = filter_for_valid_edges(valid_edges, edge_prob_df)

    
    if filtered.empty:
        return None  # or raise an exception if that’s unexpected

    # Select edge with max probability
    best_rows = filtered.nlargest(top_n, 'reward')
    return list(zip(best_rows['edge'], best_rows['pred_val_prob'],best_rows['reward'], best_rows['precedent_prob'], best_rows['t_cost']))


def best_first_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model,ml_config, top_n=1, **_):
    '''Selects the edge with the highest probability of reducing the objective value'''
    prob_df = predictor(orig_salbp, G_max_red, ml_model,ml_config)
    best_edges = select_best_n_edges(prob_df, edges,top_n)
    return best_edges
