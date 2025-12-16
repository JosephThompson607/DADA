import time
from dataset_prep import geo_from_albp_dict
def nn_graph_regressor_pred(model,salbp_instance, x_cols, edge_cols, y_graph=None, graph_label_cols=None, edge_label_df = None, salbp_type="salbp_1", cap_constraint=None, G_max_red=None, G_max_close=None, n_random=100, n_edge_random=100, feature_types={"all"}, return_assignments = False):
    ''''returns the estimated objective value for a given instance using a gnn '''
    start = time.time()
    my_geo = geo_from_albp_dict(salbp_instance, x_cols, edge_cols, y_graph=y_graph, graph_label_cols=graph_label_cols, edge_label_df = edge_label_df, salbp_type=salbp_type, cap_constraint=cap_constraint, G_max_red=G_max_red, G_max_close=G_max_close, n_random=n_random, n_edge_random=n_edge_random, feature_types=feature_types, return_assignments = return_assignments)
    output = model(my_geo)
    obj = output.item()
    elapsed_time = time.time()-start
    return {'n_stations':obj, 'elapsed_time':elapsed_time}




def nn_edge_classification_pred(model,salbp_instance, x_cols, edge_cols, y_graph=None, graph_label_cols=None, edge_label_df = None, salbp_type="salbp_1", cap_constraint=None, G_max_red=None, G_max_close=None, n_random=100, n_edge_random=100, feature_types={"all"}, return_assignments = False):
    ''''returns the estimated objective value for a given instance using a gnn '''
    start = time.time()
    my_geo = geo_from_albp_dict(salbp_instance, x_cols, edge_cols, y_graph=y_graph, graph_label_cols=graph_label_cols, edge_label_df = edge_label_df, salbp_type=salbp_type, cap_constraint=cap_constraint, G_max_red=G_max_red, G_max_close=G_max_close, n_random=n_random, n_edge_random=n_edge_random, feature_types=feature_types, return_assignments = return_assignments)
    output = model(my_geo)
    obj = output.item()
    elapsed_time = time.time()-start
    return {'n_stations':obj, 'elapsed_time':elapsed_time}