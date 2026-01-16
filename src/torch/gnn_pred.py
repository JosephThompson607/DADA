import time
from dataset_prep import geo_from_albp_dict
import yaml
from gnns import *
import torch
from train_loop import load_checkpoint, map_features_to_available, select_gnn_features,get_features, do_datasets, extract_feature_tensors
from torch_geometric.transforms import NormalizeFeatures
from alb_instance_compressor import *
from torch_geometric.data import Batch

class NNModel():
    '''Holds information neural network needs to predic'''
    def __init__(self,feature_fp, model_cp_fp,model_name,y_graph=None, graph_label_cols=None, 
                 n_random=100, n_edge_random=None, w_regression_data=True, 
                 n_hidden=32, pooling='mean', n_heads=None):
        self.feature_fp = feature_fp
        x_feat, edge_features, node_features, graph_features = get_features(self.feature_fp, w_regression_data)
        self.model = self.load_model(model_cp_fp,model_name,n_hidden, x_feat, edge_features, node_features, graph_features, pooling,n_heads)
        self.model.eval()
        self.x_cols = x_feat
        self.feature_types = self.get_feature_types(feature_fp) #This tells what features that need to be generated
        self.node_cols = node_features
        self.graph_cols = graph_features
        self.edge_cols = edge_features
        self.graph_label_cols = graph_label_cols
        self.y_graph = y_graph
        self.grap_label_cols=None
        self.n_random = n_random
        self.n_edge_random = n_edge_random
        self.w_regression_data = w_regression_data
        self.graph_dim = len(graph_features)
        self.node_dim = len(node_features)
        self.graph_ind = None
        self.node_ind = None


    def load_model(self,model_cp_fp, model, hidden_channels, x_features, edge_features, node_features, graph_features, pooling="mean", heads=None, out_channels=1):
        device="cpu"
        checkpoint = torch.load(model_cp_fp,map_location=torch.device(device), weights_only=False)
        example_data = checkpoint['sample_data'].get_example(0)


 
        edge_channels = example_data.edge_attr.size()[1]
        node_tensor, graph_tensor, node_indices, graph_indices = extract_feature_tensors(example_data.x, example_data.x_cols, node_features, graph_features)
        graph_channels = graph_tensor.size()[1]
        in_channels = node_tensor.size()[1]


        #Gets the architecture
        if model =="MLP":
            in_channels = example_data.x.size()[1] 
            model = GraphRegressorMLP(in_channels, hidden_channels, out_channels, edge_dim=edge_channels, pooling=pooling).to(device)
        elif model == "GAT":
            model =  GraphGATClassifier(in_channels, hidden_channels, out_channels, edge_dim=edge_channels, heads=heads).to(device)
        elif model == "GAT3":
            model =  GraphGATClassifier3Layer(in_channels, hidden_channels, out_channels, edge_dim=edge_channels,heads=heads).to(device)
        elif model == "GCN":
            model = GraphClassifier(in_channels, hidden_channels, out_channels).to(device)
        elif model == "GCN3":
            model = GraphClassifier3Layer(in_channels, hidden_channels, out_channels).to(device)
        elif model == "GATStats":
            model =  GraphGATClassifierStats(in_channels,graph_channels, node_indices, graph_indices, hidden_channels, out_channels, edge_dim = edge_channels, heads=heads, pooling=pooling).to(device)
        elif model == "GAT3Stats":
            model = GraphGAT3LayerClassifierStats(in_channels,graph_channels, node_indices, graph_indices, hidden_channels, out_channels, edge_dim = edge_channels, heads=heads, pooling=pooling).to(device)
        elif model == "GCNStats":
            model = GraphClassifierStats(in_channels,graph_channels, node_indices, graph_indices, hidden_channels, out_channels, pooling=pooling).to(device)
        elif model == "GCN3Stats":
            model = GraphClassifier3LayerStats(in_channels,graph_channels, node_indices, graph_indices, hidden_channels, out_channels, pooling=pooling).to(device)
        #Loads weights for model
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


    def get_feature_types(self, feature_fp):
        with open(feature_fp, 'r') as file:
            data = yaml.safe_load(file)

        # Extract the value
        return set(data['feature_types'])


    def graph_regressor_pred(self,salbp_instance, cap_constraint = None,G_max_red=None, G_max_close=None,  return_assignments = False):
        ''''returns the estimated objective value for a given instance dict. '''
        start = time.time()
        my_geo = geo_from_albp_dict(salbp_instance, self.x_cols, self.edge_cols, y_graph=self.y_graph,
                                    cap_constraint=cap_constraint, G_max_red=G_max_red, G_max_close=G_max_close, 
                                    n_random=self.n_random, n_edge_random=self.n_edge_random, 
                                    feature_types=self.feature_types, 
                                    return_assignments = return_assignments)
        
        print(my_geo)
        print('geo cols ', my_geo.x_cols)
        # Create the same transform
        # transform = NormalizeFeatures()
        # data_new = transform(my_geo)

        if not hasattr(my_geo, 'batch') or my_geo.batch is None:
            my_geo = Batch.from_data_list([my_geo])
        output = self.model(my_geo)
        obj = output.item()
        elapsed_time = time.time()-start
        return {'n_stations':obj, 'elapsed_time':elapsed_time}


        

# def nn_graph_regressor_pred(model,salbp_instance, x_cols, edge_cols, y_graph=None, graph_label_cols=None, edge_label_df = None, salbp_type="salbp_1", cap_constraint=None, G_max_red=None, G_max_close=None, n_random=100, n_edge_random=100, feature_types={"all"}, return_assignments = False):
#     ''''returns the estimated objective value for a given instance using a gnn '''
#     start = time.time()
#     my_geo = geo_from_albp_dict(salbp_instance, x_cols, edge_cols, y_graph=y_graph, graph_label_cols=graph_label_cols, edge_label_df = edge_label_df, salbp_type=salbp_type, cap_constraint=cap_constraint, G_max_red=G_max_red, G_max_close=G_max_close, n_random=n_random, n_edge_random=n_edge_random, feature_types=feature_types, return_assignments = return_assignments)
#     output = model(my_geo)
#     obj = output.item()
#     elapsed_time = time.time()-start
#     return {'n_stations':obj, 'elapsed_time':elapsed_time}

# #nn_edge_classification_pred( orig_salbp, G_max_close, G_max_red, ml_model, ml_config)




# def nn_edge_classification_pred(model,salbp_instance, G_max_red=None, G_max_close=None,feature_types={"all"}, return_assignments = False):
#     ''''returns the estimated objective value for a given instance using a gnn '''
#     start = time.time()
#     my_geo = geo_from_albp_dict(salbp_instance,  G_max_red=G_max_red, G_max_close=G_max_close, feature_types=feature_types, return_assignments = return_assignments)
#     output = model.model(my_geo)
#     obj = output.item()
#     elapsed_time = time.time()-start
#     return {'n_stations':obj, 'elapsed_time':elapsed_time}



def main():
    feature_fp = "/home/jot240/DADA/DADA/data/ml_models/hyper_parameter/features/edge_no_rw_no_edgeeval.yaml"
    # hidden_channels = 128
    # cp_fp = "/home/jot240/scratch/pytorch_checkpoints/regression_tuning_STATS_take2/trial_2_20251213_011943/best_model.pt"
    # model_name = "GCN3Stats"
    

    # hidden_channels = 64
    # cp_fp = "/home/jot240/scratch/pytorch_checkpoints/regression_tuning_STATSMLP_take1/trial_3_20251216_144308/best_model.pt"
    # model_name = "MLP"
    # gnn_regresssor = NNModel(feature_fp, cp_fp, model_name, n_hidden=hidden_channels)


    hidden_channels = 64
    cp_fp = "/home/jot240/scratch/pytorch_checkpoints/regression_tuning_STATSMLP_take1/trial_5_20251217_084400/best_model.pt"
    model_name = "GCN3Stats"
    gnn_regresssor = NNModel(feature_fp, cp_fp, model_name, n_hidden=hidden_channels)
   

    alb_dicts = open_salbp_pickle("/home/jot240/DADA/DADA/data/raw/pkl_datasets/otto/large.pkl")
    salbp_prob = alb_dicts[0]

    print("hhere is the problem", salbp_prob)
    sol = gnn_regresssor.graph_regressor_pred(salbp_prob)
    print("here is theh solution", sol)



if __name__ == "__main__":
    main()
