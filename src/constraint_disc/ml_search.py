import pandas as pd
import sys

sys.path.append('src')
from data_prep import albp_to_features
import copy

def predictor(orig_salbp, G_max_red, ml_model, ml_config, n_random=0,**kwargs):
    #Take task times form base salbp problem and use them with the new edges
    ml_feature_list = ml_config["features"]
    test_salbp = copy.deepcopy(orig_salbp)
    test_salbp['precedence_relations'] = list(G_max_red.edges())
    edge_features = albp_to_features(test_salbp, salbp_type="salbp_1", cap_constraint = None, n_random=n_random, feature_types=set(ml_config['feature_types']))
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
    def get_edge_prob(edge):
        # edge is assumed to be a tuple (u, v)
        return G_max_red.edges[edge].get('prob', 1) if edge in G_max_red.edges else 1

    edge_prob_df['precedent_prob'] = edge_prob_df['edge'].apply(get_edge_prob)
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
    best_row = filtered.loc[filtered['pred_val_prob'].idxmax()]
    return best_row['edge'], best_row['pred_val_prob']

def select_best_n_edges(edge_prob_df, valid_edges, top_n):
    """
    If given an edge probability feature, we multiply the ml value by the probability of the edge not existing. We then Filter edge_prob_df to only include edges in valid_edges,
    then return the edge with the highest predicted probability.
    """
    
    edge_prob_df['reward'] =  edge_prob_df['precedent_prob']* edge_prob_df['pred_val_prob']

    # Filter DataFrame by valid edges
   
    valid_edges = set([(str(e1),str(e2)) for (e1,e2) in valid_edges])
    edge_prob_df['valid'] = edge_prob_df['edge'].apply(lambda x: (str(x[0]), str(x[1])) in valid_edges)
    filtered = edge_prob_df[edge_prob_df['valid']==True]
    if filtered.empty:
        print("Error: No edges in the probability dataframe ")
        return None  # or raise an exception if that’s unexpected
    
    
    # Select edge with max probability
    best_rows = filtered.nlargest(top_n, 'reward')
    return list(zip(best_rows['edge'], best_rows['reward'], best_rows['precedent_prob']))


def best_first_ml_choice_edge(edges, orig_salbp, G_max_red, ml_model,top_n=1, prob_weights=None, **kwargs):
    '''Selects the edge with the highest probability of reducing the objective value'''
    prob_df = predictor(orig_salbp, G_max_red, ml_model, **kwargs)
    best_edges = select_best_n_edges(prob_df, edges,top_n, prob_weights)

    return best_edges
