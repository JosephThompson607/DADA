import time
from dataset_prep import geo_from_albp_dict
from train_loop import get_features


class ConfigNN():
    '''Holds information neural network needs to predic'''
    def __init__(self,feature_fp, model,y_graph=None, graph_label_cols=None, n_random=None, n_edge_random=None, w_regression_data=False):
        self.feature_fp = feature_fp
        self.model = model
        x_feat, edge_features, node_features, graph_features = get_features(self.feature_fp, w_regression_data=self.w_regression_data)
        self.x_cols = x_feat
        self.node_cols = node_features,
        self.graph_cols = graph_features,
        self.edge_cols = edge_features,
        self.graph_label_cols = graph_label_cols,
        self.y_graph = y_graph
        self.grap_label_cols=None,
        self.n_random = n_random,
        self.n_edge_random = n_edge_random,
        self.w_regression_data = w_regression_data
        self.graph_dim = len(graph_features)
        self.node_dim = len(node_features)
        self.graph_ind = None,
        self.node_ind = None,


        

def nn_graph_regressor_pred(model,salbp_instance, x_cols, edge_cols, y_graph=None, graph_label_cols=None, edge_label_df = None, salbp_type="salbp_1", cap_constraint=None, G_max_red=None, G_max_close=None, n_random=100, n_edge_random=100, feature_types={"all"}, return_assignments = False):
    ''''returns the estimated objective value for a given instance using a gnn '''
    start = time.time()
    my_geo = geo_from_albp_dict(salbp_instance, x_cols, edge_cols, y_graph=y_graph, graph_label_cols=graph_label_cols, edge_label_df = edge_label_df, salbp_type=salbp_type, cap_constraint=cap_constraint, G_max_red=G_max_red, G_max_close=G_max_close, n_random=n_random, n_edge_random=n_edge_random, feature_types=feature_types, return_assignments = return_assignments)
    output = model(my_geo)
    obj = output.item()
    elapsed_time = time.time()-start
    return {'n_stations':obj, 'elapsed_time':elapsed_time}

#nn_edge_classification_pred( orig_salbp, G_max_close, G_max_red, ml_model, ml_config)




def nn_edge_classification_pred(model,salbp_instance, G_max_red=None, G_max_close=None,feature_types={"all"}, return_assignments = False):
    ''''returns the estimated objective value for a given instance using a gnn '''
    start = time.time()
    my_geo = geo_from_albp_dict(salbp_instance,  G_max_red=G_max_red, G_max_close=G_max_close, feature_types=feature_types, return_assignments = return_assignments)
    output = model.model(my_geo)
    obj = output.item()
    elapsed_time = time.time()-start
    return {'n_stations':obj, 'elapsed_time':elapsed_time}