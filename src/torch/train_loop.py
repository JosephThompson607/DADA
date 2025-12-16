import torch
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import graphviz
import torch_geometric
import sys
import os
import shutil
from types import SimpleNamespace
src_path = os.path.abspath("src")
if src_path not in sys.path:
    sys.path.append(src_path)
src_path = os.path.abspath("src/torch")
if src_path not in sys.path:
    sys.path.append(src_path)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from SALBP_solve import *
import os.path as osp
import ast
from sklearn.preprocessing import MinMaxScaler
import torchmetrics
from torch.utils.data import random_split,ConcatDataset
from datetime import datetime
sys.path.append(os.path.abspath("torch"))
from salb_dataset import *
from gnns import *

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model and optimizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('eval_loss', [])
    sample_data = checkpoint.get('sample_data',None)
    if sample_data is not None:
        sample_data.to(device)
    print(f'Loaded checkpoint from epoch {epoch}')
    return epoch, train_losses, test_losses, sample_data


def train_with_checkpoints(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    checkpoint_dir='checkpoints',
    save_every=10,
    save_best=True,
    mode = 'regressor'
):
    """
    Train model with periodic checkpoint saving.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        epochs: Number of epochs
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        save_best: Whether to save best model based on validation loss
    
    Returns:
        dict with 'train_losses' and 'test_losses'
    """
    # Create checkpoint directory

    
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        total_loss = 0
        model.train()
        sample_data = next(iter(train_loader))
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            if mode == 'regressor':
                loss = loss_fn(out.squeeze(1), data.y.float())
            elif mode == 'classifier':
                loss = loss_fn(out.squeeze(1), data.y_edge[:,0].float())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}')
        
        # Evaluation
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                if mode == 'regressor':
                    loss = loss_fn(out.squeeze(1), data.y.float())
                elif mode == 'classifier':
                    loss = loss_fn(out.squeeze(1), data.y_edge[:,0].float())

                total_loss += loss.item()
        
        eval_loss = total_loss / len(test_loader)
        test_losses.append(eval_loss)
        print(f'Epoch {epoch+1}/{epochs}, Eval Loss: {eval_loss:.4f}')
        
        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_eval_loss_{eval_loss:.2f}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'sample_data':sample_data.to("cpu"),
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
        
        # Save best model
        if save_best and eval_loss < best_loss:
            best_loss = eval_loss
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'sample_data':sample_data.to("cpu")
            }, best_path)
            print(f'Saved best model with eval loss: {eval_loss:.4f}')
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'sample_data':sample_data.to("cpu")
    }, final_path)
    print(f'Saved final model: {final_path}')
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses
    }



def extract_feature_tensors(data_x, x_cols, node_features, graph_features):
    """
    Extract node and graph feature tensors from data.x
    
    Args:
        data_x: PyTorch tensor of shape (num_nodes, num_features)
        x_cols: List of all feature names corresponding to data_x columns
        node_features: List of node feature names to extract
        graph_features: List of graph feature names to extract
    
    Returns:
        tuple: (node_tensor, graph_tensor, node_indices, graph_indices)
    """
    # Get indices
    node_indices = [x_cols.index(f) for f in node_features if f in x_cols]
    graph_indices = [x_cols.index(f) for f in graph_features if f in x_cols]
    
    # Convert to torch tensors for advanced indexing
    node_idx_tensor = torch.tensor(node_indices, dtype=torch.long)
    graph_idx_tensor = torch.tensor(graph_indices, dtype=torch.long)
    
    # Slice tensors
    node_tensor = data_x[:, node_idx_tensor] if len(node_indices) > 0 else None
    graph_tensor = data_x[:, graph_idx_tensor] if len(graph_indices) > 0 else None
    
    return node_tensor, graph_tensor, node_indices, graph_indices


def do_datasets(node_feature_list = [], edge_feature_list = []):
    n_range =  [50,60,90, 100]
    #n_range =  [50,]
    datasets = ['unstructured', 'chains', 'bottleneck']
    ds_list = []
    print(f"In do datatsets, selecting {node_feature_list} \n\n {edge_feature_list}")
    for n in n_range:
        for ds in datasets:
            print(f"processing{n}_{ds}")
            pkl_fp = f"/home/jot240/DADA/DADA/data/pereria_results/pytorch_ready/{ds}_n_{n}_geo_ready.pkl"
            root_fp = f"/home/jot240/DADA/DADA/data/pytorch_datasets/regression/"
            dataset_name =f'{ds}_n_{n}'
            unstructured = create_and_load_salbp_dataset(root_fp, pkl_fp, dataset_name)
            if len(node_feature_list)>0:
                sliced = unstructured.select_features(node_feature_list, edge_feature_list)
                ds_list.append(sliced)
            else:
                ds_list.append(unstructured)
    my_dataset = ConcatDataset(ds_list)
    return my_dataset



def setup_and_train( hidden_channels,  learning_rate, epochs, heads, batch_size, model, checkpoint_dir, save_every=20,x_features = [],edge_features=[],node_features=[], graph_features=[], seed=None, pooling='mean'):
    my_dataset = do_datasets(x_features, edge_features)
    in_channels = my_dataset[0].x.size()[1] 
    edge_channels = my_dataset[0].edge_attr.size()[1]
    print(f"size of datset: {len(my_dataset)} dataset splitting seed {seed}" )
    out_channels = 1 
    #splits the data into train and test
    transform = NormalizeFeatures()
    my_dataset.transform = transform
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(my_dataset, [int(len(my_dataset)*0.8), len(my_dataset) - int(len(my_dataset)*0.8)], generator=generator)
    else:
        train_dataset, test_dataset = random_split(my_dataset, [int(len(my_dataset)*0.8), len(my_dataset) - int(len(my_dataset)*0.8)])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:", device, " Setting model: ", model)
    if "Stats" in model:
        #Get node level and graph level feature indicies for slicing x array.
        node_tensor, graph_tensor, node_indices, graph_indices = extract_feature_tensors(my_dataset[0].x, my_dataset[0].x_cols, node_features, graph_features)
        graph_channels = graph_tensor.size()[1]
        in_channels = node_tensor.size()[1]
    if model =="MLP":
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

    else:
        print(f"Error: GNN Architecture '{model}' does not exist.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    losses_dict = train_with_checkpoints(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    checkpoint_dir=checkpoint_dir,
    save_every=save_every,
    save_best=True
)
    return losses_dict

def do_datasets_classifier(node_feature_list = [], edge_feature_list = []):
    n_range =  [50,60,90, 100]
    #n_range =  [50,]
    datasets = ['unstructured', 'chains', 'bottleneck']
    #datasets = ['unstructured']
    ds_list = []
    print(f"In do datatsets, selecting {node_feature_list} \n\n  edge features: {edge_feature_list}")
    for n in n_range:
        for ds in datasets:
            print(f"processing{n}_{ds}")
            pkl_fp = f"/home/jot240/DADA/DADA/data/results/{ds}_{n}/{ds}_n_{n}_geo_ready_edge_res.pkl"
            root_fp = f"/home/jot240/DADA/DADA/data/pytorch_datasets/classification/"
            dataset_name =f'{ds}_n_{n}'
            unstructured = create_and_load_salbp_dataset(root_fp, pkl_fp, dataset_name)
            if len(node_feature_list)>0:
                sliced = unstructured.select_features(node_feature_list, edge_feature_list)
                ds_list.append(sliced)
            else:
                ds_list.append(unstructured)
    my_dataset = ConcatDataset(ds_list)
    print("DATASET SHAPES", my_dataset[0])
    return my_dataset

def get_pos_weight(data_list):
    
    num_pos = sum(data.y_edge[:,0].sum() for data in data_list)
    num_neg = sum(len(data.y_edge) - data.y_edge[0].sum().item() for data in data_list)
    
    # Compute pos_weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([num_neg / num_pos])  # Shape: (1,)
    print(f"num pos {num_pos} num neg {num_neg} pos weight {pos_weight}")
    percent_pos = num_pos/(num_pos + num_neg)
    percent_neg = num_neg/(num_pos + num_neg)
    print(f"Percent positive {percent_pos} percent negative {percent_neg}")
    return pos_weight

def setup_and_train_classifier( hidden_channels,  learning_rate, epochs, heads, batch_size, model, checkpoint_dir, save_every=20,x_features = [],node_features=[],edge_features=[],graph_features=[], seed=None):
    my_dataset = do_datasets_classifier(x_features, edge_features)
    print(f"size of datset: {len(my_dataset)} dataset splitting seed {seed}" )
    
    in_channels = my_dataset[0].x.size()[1] 
    edge_channels = my_dataset[0].edge_attr.size()[1]
    out_channels = 1 
    #splits the data into train and test
    transform = NormalizeFeatures()
    my_dataset.transform = transform
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(my_dataset, [int(len(my_dataset)*0.8), len(my_dataset) - int(len(my_dataset)*0.8)], generator=generator)
    else:
        train_dataset, test_dataset = random_split(my_dataset, [int(len(my_dataset)*0.8), len(my_dataset) - int(len(my_dataset)*0.8)])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:", device, " Setting model: ", model)
    if model =="MLP":
        model = EdgeClassifierMLP(in_channels, hidden_channels, out_channels, edge_dim=edge_channels).to(device)

    elif model == "GAT":
        model = EdgeClassifierGAT(in_channels, hidden_channels, out_channels, edge_dim = edge_channels, heads=heads).to(device)
    elif model == "GAT3":
        model = EdgeClassifierGAT3Layer(in_channels, hidden_channels, out_channels, edge_dim = edge_channels, heads=heads).to(device)
    elif model == "GCN":
        print("using GCN with 2 conv, 2 fully connected layers")
        model = EdgeClassifier(in_channels, hidden_channels, out_channels, edge_dim=edge_channels).to(device)
    elif model == "GCN3":
        print("using GCN with 3 conv, 3 fully connected layers")
        model = EdgeClassifier3Conv3Lin(in_channels, hidden_channels, out_channels, edge_dim=edge_channels).to(device)
    else:
        print(f"Error: GNN Architecture '{model}' does not exist.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #Setting up classifier for imbalanced training set
    pos_weight = get_pos_weight(my_dataset)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    losses_dict = train_with_checkpoints(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    checkpoint_dir=checkpoint_dir,
    save_every=save_every,
    save_best=True,
    mode = 'classifier'
)
    return losses_dict



def map_features_to_available(feature_list, available_edge, available_x, available_graph, w_regression_data):
    """
    Maps features from feature_list to available_edge and available_x.
    
    Special mappings:
    - 'parent_weight' -> 'value' (in available_x)
    - 'child_weight' -> 'value' (in available_x)
    - Features with 'parent_' prefix map to available_x features without prefix
    - Features with 'child_' prefix map to available_x features without prefix
    
    Returns:
        dict: {'edge_features': [...], 'node_features': [...], 'mappings': {...}}
    """
    edge_features = []
    node_features = []
    x_features = []
    graph_features = []
    mappings = {}
    
    # Special mappings
    special_maps = {
        'parent_weight': 'value',
        'child_weight': 'value',
        'parent_pos_weight': 'pos_weight',
        'child_pos_weight': 'pos_weight',
        'load_parent_mean': 'load_mean',
        'load_parent_max':  'load_max',
        'load_parent_min':  'load_min',
        'load_parent_std':  'load_std',
        'load_child_mean': 'load_mean',
        'load_child_max':  'load_max',
        'load_child_min':  'load_min',
        'load_child_std':  'load_std',
    }
    
    for feature in feature_list:
        # Check for exact match in edge features
        if feature in available_edge:
            edge_features.append(feature)
            mappings[feature] = ('edge', feature)
            continue
        
        # Check for exact match in node features
        if feature in available_x:
            x_features.append(feature)
            mappings[feature] = ('x', feature)
            
            if feature in available_graph:
                graph_features.append(feature)
                mappings[feature] = ('graph', feature)
            else:
                node_features.append(feature)
                mappings[feature] = ('node', feature)
            continue
            
        # Check special mappings
        if feature in special_maps:
            target = special_maps[feature]
            if target in available_x:
                x_features.append(target)
                node_features.append(target)
                mappings[feature] = ('node', target, 'special_mapping')
                continue
        
        # Check for parent_* -> * mapping
        if feature.startswith('parent_'):
            suffix = feature[7:]  # Remove 'parent_' prefix
            if suffix in available_x:
                x_features.append(suffix)
                node_features.append(suffix)
                mappings[feature] = ('node', suffix, 'parent_mapping')
                continue
        
        # Check for child_* -> * mapping
        if feature.startswith('child_'):
            suffix = feature[6:]  # Remove 'child_' prefix
            if suffix in available_x:
                x_features.append(suffix)
                node_features.append(suffix)
                mappings[feature] = ('node', suffix, 'child_mapping')
                continue
        
        # Feature not found
        mappings[feature] = ('not_found', None)
    if w_regression_data:
        to_append = ['lb_6']
        x_features += to_append
        graph_features += to_append
        substring = "priority"
        if any(substring in s for s in graph_features):
            load_stats = ['priority_min_stations', 'priority_max_stations']
            graph_features+= load_stats
            x_features += load_stats
    # Remove duplicates while preserving order
    edge_features = list(dict.fromkeys(edge_features))
    node_features = list(dict.fromkeys(node_features))
    graph_features = list(dict.fromkeys(graph_features))
    x_features = list((dict.fromkeys(x_features)))
    if "NO_EDGE_FEATURES" in feature_list:
        print("NOTE: NO EDGE FEATURES WILL BE USED, 'NO_EDGE_FEATURES' found in feature list")
        edge_features=None
    return x_features,edge_features,node_features, graph_features,mappings
    

def select_gnn_features(feat_list, w_regression_data):
    available_edge = ['child_min',
                     'child_max',
                     'child_avg',
                     'child_std',
                     'parent_min',
                     'parent_max',
                     'parent_avg',
                     'parent_std',
                     'neighborhood_min',
                     'neighborhood_max',
                     'neighborhood_avg',
                     'neighborhood_std',
                     'abs_weight_difference',
                     'stage_difference',
                     'chain_avg',
                     'chain_min',
                     'chain_max',
                     'chain_std',
                     'stations_delta',
                     'weight_sum']
    available_x = ['value',
                     'pos_weight',
                     'stage',
                     'in_degree',
                     'out_degree',
                     'load_mean',
                     'load_max',
                     'load_min',
                     'load_std',
                     'rw_mean_total_time',
                     'rw_mean_min_time',
                     'rw_mean_max_time',
                     'rw_mean_n_unique_nodes',
                     'rw_mean_walk_length',
                     'rw_min',
                     'rw_max',
                     'rw_mean',
                     'rw_std',
                     'rw_n_unique_nodes',
                     'priority_min_stations',
                     'priority_max_stations',
                     'priority_min_gap',
                     'priority_max_gap',
                     'random_spread',
                     'random_coefficient_of_variation',
                     'random_avg_gap',
                     'random_min_gap',
                     'random_max_gap',
                     'random_avg_efficiency',
                     'min_div_c',
                     'max_div_c',
                     'sum_div_c',
                     'std_div_c',
                     't_cv',
                     'ti_size',
                     'avg_div_c',
                     'lb_6',
                     'n_edges',
                     'order_strength',
                     'average_number_of_immediate_predecessors',
                     'max_degree',
                     'max_in_degree',
                     'max_out_degree',
                     'divergence_degree',
                     'convergence_degree',
                     'n_bottlenecks',
                     'share_of_bottlenecks',
                     'avg_degree_of_bottlenecks',
                     'n_chains',
                     'avg_chain_length',
                     'nodes_in_chains',
                     'n_stages',
                     'stages_div_n',
                     'prec_strength',
                     'prec_bias',
                     'prec_index',
                     'n_isolated_nodes',
                     'share_of_isolated_nodes',
                     'n_tasks_without_predecessors',
                     'share_of_tasks_without_predecessors',
                     'avg_tasks_per_stage']
    available_graph = [
                     'priority_min_stations',
                     'priority_max_stations',
                     'priority_min_gap',
                     'priority_max_gap',
                     'random_spread',
                     'random_coefficient_of_variation',
                     'random_avg_gap',
                     'random_min_gap',
                     'random_max_gap',
                     'random_avg_efficiency',
                     'min_div_c',
                     'max_div_c',
                     'sum_div_c',
                     'std_div_c',
                     't_cv',
                     'ti_size',
                     'avg_div_c',
                     'lb_6',
                     'n_edges',
                     'order_strength',
                     'average_number_of_immediate_predecessors',
                     'max_degree',
                     'max_in_degree',
                     'max_out_degree',
                     'divergence_degree',
                     'convergence_degree',
                     'n_bottlenecks',
                     'share_of_bottlenecks',
                     'avg_degree_of_bottlenecks',
                     'n_chains',
                     'avg_chain_length',
                     'nodes_in_chains',
                     'n_stages',
                     'stages_div_n',
                     'prec_strength',
                     'prec_bias',
                     'prec_index',
                     'n_isolated_nodes',
                     'share_of_isolated_nodes',
                     'n_tasks_without_predecessors',
                     'share_of_tasks_without_predecessors',
                     'avg_tasks_per_stage']
    x_features,edge_features,node_features, graph_features,mappings = map_features_to_available(feat_list, available_edge, available_x, available_graph, w_regression_data)
    print("Edge Features:", edge_features)
    print("\X Features:", x_features)
    print("\nNode Features(also lumped in with x):", node_features)
    print("\nGraph Features (also lumped in with x):", graph_features)
    print("\nNot Found:")
    for feature, mapping in mappings .items():
        if mapping[0] == 'not_found':
            print(f"  - {feature}")
            
    print("\nSpecial Mappings:")
    for feature, mapping in mappings.items():
        if len(mapping) > 2:
            print(f"  - {feature} -> {mapping[1]} ({mapping[2]})")
    return x_features, edge_features,node_features,graph_features

def load_configuration(feature_fp):
    print("using features specified in ", feature_fp)
    with open(feature_fp, 'r') as file:
        data = yaml.safe_load(file)
        features_list = data['features']
    return features_list       


def get_features(feature_fp, w_regression_data=False):
    feat_list = load_configuration(feature_fp)
    x_feat, edge_features, node_features, graph_features = select_gnn_features(feat_list, w_regression_data)
    return x_feat, edge_features, node_features, graph_features


def copy_config_to_output(config_path, output_dir):
    if config_path is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, os.path.basename(config_path))
    shutil.copy(config_path, dest_path)
    print(f"Copied config file → {dest_path}")

def load_yaml_config(path):
    """Load YAML config if provided."""
    if path is None:
        return {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file '{path}' not found")

    print(f"Loading configuration from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def merge_config_with_cli(config_dict, cli_args):
    """
    Merge config file values with CLI args.
    CLI overrides config.
    Returns an object that behaves like argparse.Namespace
    """

    merged = {}

    # First put config values
    for k, v in config_dict.items():
        merged[k] = v

    # Then override with CLI args if they are not None
    for k, v in vars(cli_args).items():
        if v is not None:
            merged[k] = v
    if cli_args.additional_features is not None:
        print(f"adding features {cli_args.additional_features} from command line")
        merged['features'] = merged['features'] + cli_args.additional_features
    return SimpleNamespace(**merged)
def main():
    parser = argparse.ArgumentParser(description="Train GNN with optional YAML config")

    # Add CLI args
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file")

    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data_seed", type=int, default=None, help="seed for train test split")
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--pooling", type=str, default='mean', help="What pooling layer to use for graph regression")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--features", type=str, nargs="*", default=None,
                        help="Optional list of features OR put in config file")
    parser.add_argument("--additional_features", type=str, nargs="*", default=None,
                        help="Optional list of features that is appended to config file")

    args = parser.parse_args()

    # ----------------------------------------------------
    # 1. Load config file (if provided)
    # ----------------------------------------------------
    config_dict = load_yaml_config(args.config)

    # ----------------------------------------------------
    # 2. Merge config values with CLI (CLI wins)
    # ----------------------------------------------------
    cfg = merge_config_with_cli(config_dict, args)
    today = datetime.today().strftime("%Y-%m-%d")
    cfg.checkpoint_dir += today
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print("\n==== Final Training Configuration (config + CLI override) ====")
    print(cfg)
    print("=============================================================\n")

    # ----------------------------------------------------
    # 3. Copy config file to output dir
    # ----------------------------------------------------
    if cfg.checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be defined in config or CLI")

    copy_config_to_output(args.config, cfg.checkpoint_dir)

    # ----------------------------------------------------
    # 4. Feature extraction logic
    # ----------------------------------------------------
    #Adds in regression specific features
    w_regression_data = False
    if cfg.model_type == "graph_regression":
        w_regression_data=True
    if "features" in config_dict and config_dict["features"] is not None:
        
        x_features, edge_features, node_features, graph_features = select_gnn_features(cfg.features, w_regression_data)
    
    elif args.features is not None:
        x_features, edge_features, node_features, graph_features= select_gnn_features(args.features,w_regression_data)
    else:
        node_features, edge_features = [],None
        print("No features specified; using all features")

    # ----------------------------------------------------
    # 5. Dispatch model type
    # ----------------------------------------------------
    if cfg.model_type == "graph_regression":
        setup_and_train(cfg.hidden_channels, cfg.learning_rate, cfg.epochs,
                        cfg.heads, cfg.batch_size, cfg.architecture,
                        cfg.checkpoint_dir, x_features=x_features, edge_features=edge_features, seed=args.data_seed, graph_features=graph_features, node_features=node_features, pooling=cfg.pooling)

    elif cfg.model_type == "edge_classification":
        setup_and_train_classifier(cfg.hidden_channels, cfg.learning_rate, cfg.epochs,
                                   cfg.heads, cfg.batch_size, cfg.architecture,
                                   cfg.checkpoint_dir, x_features=x_features, edge_features=edge_features, graph_features=graph_features, seed=args.data_seed)

    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

if __name__ == "__main__":
    main()





