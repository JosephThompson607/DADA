
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch.utils.data import random_split
from torch_geometric.utils import scatter
class EdgeClassifier(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super(EdgeClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        edge_input_dim = 2 * hidden_channels

            
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data,**data_kwargs):
        x = data.x
        edge_index = data.edge_index
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Edge embedding - gather node features for each edge
        parents, children = edge_index
        edge_features = torch.cat([x[parents], x[children]], dim=1)
        
        # Edge classification
        return self.edge_mlp(edge_features)
    
    
class EdgeClassifier_GAT(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(EdgeClassifier_GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)  # TODO
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=0.6)  # TODO
        edge_input_dim = 2 * hidden_channels

            
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self,data,**data_kwargs):
        x = data.x
        edge_index = data.edge_index
        # Node embedding
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Edge embedding - gather node features for each edge
        parents, children = edge_index
        edge_features = torch.cat([x[parents], x[children]], dim=1)
        
        # Edge classification
        return self.edge_mlp(edge_features)
    
class EdgeClassifierGATStats(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(EdgeClassifierGATStats, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)  # TODO
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=0.6)  # TODO
        edge_input_dim = 2 * hidden_channels + 3 #The 3 is from the graph statistics we are calculating
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data,**data_kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Apply GAT convolutions to get node embeddings
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        node_embeddings = self.conv2(x, edge_index)
        
        # Calculate graph statistics from original node features
        # Assuming x[:, 0] contains the x-values
        x_values = x[:, 0]
        
        # If batch information is not provided, assume single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        stats_tensor = torch.stack(
            [data_kwargs["graph_data"][fp][instance] for (fp,instance) in zip(data.dataset, data.instance)],
            dim=0
        ).to(x.device)
        # # Calculate per-graph statistics
        # mean_x = scatter(x_values, batch, dim=0, reduce="mean")
        # # For std, we need to calculate variance first and then sqrt
        # var_x = scatter((x_values - mean_x[batch])**2, batch, dim=0, reduce="mean")
        # std_x = torch.sqrt(var_x + 1e-8)  # Adding small epsilon for numerical stability
        # min_x = scatter(x_values, batch, dim=0, reduce="min")[0]
        # max_x = scatter_max(x_values, batch, dim=0, reduce="max")[0]
        
        # For each edge, get source and target node embeddings
        row, col = edge_index
        edge_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        
        # Add per-graph statistics to each edge feature
        # Determine which graph each edge belongs to
        # edge_batch = batch[row]  # Using the source node's batch assignment
        
        # Create tensor to hold graph stats for each edge
        # graph_stats = torch.cat([
        #     mean_x[edge_batch].unsqueeze(1),
        #     std_x[edge_batch].unsqueeze(1),
        #     min_x[edge_batch].unsqueeze(1),
        #     max_x[edge_batch].unsqueeze(1)
        # ], dim=1)
        
        # Concatenate edge features with graph statistics
        edge_features = torch.cat([edge_features, stats_tensor], dim=1)
        
        # Pass through edge MLP for classification
        edge_pred = self.edge_mlp(edge_features)
        
        return edge_pred
    

class EdgeClassifierGATStats3Layer(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(EdgeClassifierGATStats, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)  # TODO
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=0.6)  # TODO
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels,dropout=0.6)  # TODO
        edge_input_dim = 2 * hidden_channels + 3 #The 3 is from the graph statistics we are calculating
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data,**data_kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Apply GAT convolutions to get node embeddings
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        node_embeddings = self.conv2(x, edge_index)
        
        # Calculate graph statistics from original node features
        # Assuming x[:, 0] contains the x-values
        x_values = x[:, 0]
        
        # If batch information is not provided, assume single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        stats_tensor = torch.stack(
            [data_kwargs["graph_data"][fp][instance] for (fp,instance) in zip(data.dataset, data.instance)],
            dim=0
        ).to(x.device)
        # # Calculate per-graph statistics
        # mean_x = scatter(x_values, batch, dim=0, reduce="mean")
        # # For std, we need to calculate variance first and then sqrt
        # var_x = scatter((x_values - mean_x[batch])**2, batch, dim=0, reduce="mean")
        # std_x = torch.sqrt(var_x + 1e-8)  # Adding small epsilon for numerical stability
        # min_x = scatter(x_values, batch, dim=0, reduce="min")[0]
        # max_x = scatter_max(x_values, batch, dim=0, reduce="max")[0]
        
        # For each edge, get source and target node embeddings
        row, col = edge_index
        edge_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        
        # Add per-graph statistics to each edge feature
        # Determine which graph each edge belongs to
        # edge_batch = batch[row]  # Using the source node's batch assignment
        
        # Create tensor to hold graph stats for each edge
        # graph_stats = torch.cat([
        #     mean_x[edge_batch].unsqueeze(1),
        #     std_x[edge_batch].unsqueeze(1),
        #     min_x[edge_batch].unsqueeze(1),
        #     max_x[edge_batch].unsqueeze(1)
        # ], dim=1)
        
        # Concatenate edge features with graph statistics
        edge_features = torch.cat([edge_features, stats_tensor], dim=1)
        
        # Pass through edge MLP for classification
        edge_pred = self.edge_mlp(edge_features)
        
        return edge_pred
    
class GraphClassifier(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
    def forward(self, data,**data_kwargs):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_mlp(x)
        

        return x

class GraphClassifier3Layer(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels):
        super(GraphClassifier3Layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
    def forward(self, data,**data_kwargs):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_mlp(x)
        

        return x
    



class GraphGATClassifier(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.6):
        super(GraphGATClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads , dropout=dropout)  # TODO
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads,
                             concat=False, dropout=dropout)  # TODO
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
    def forward(self, data, **data_kwargs):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_mlp(x)
        

        return x
    



class GraphGATClassifierStats(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.6, n_features=15):
        super(GraphGATClassifierStats, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads , dropout=dropout)  # TODO
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads,
                             concat=False, dropout=dropout)  # TODO
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + n_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
    def forward(self, data, **data_kwargs):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        stats_tensor = torch.stack(
            [data_kwargs["graph_data"][fp][instance] for (fp,instance) in zip(data.dataset, data.instance)],
            dim=0
        ).to(x.device)
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x, stats_tensor], dim=1)
        x = self.graph_mlp(x)
        

        return x
    

class GraphGATClassifierStats3Layer(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.6, n_features=15):
        super(GraphGATClassifierStats3Layer, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads , dropout=dropout)  
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads, dropout=dropout)  
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads,
                             concat=False, dropout=dropout)  
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + n_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
    def forward(self, data, **data_kwargs):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        stats_tensor = torch.stack(
            [data_kwargs["graph_data"][fp][instance] for (fp,instance) in zip(data.dataset, data.instance)],
            dim=0
        ).to(x.device)
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x, stats_tensor], dim=1)
        x = self.graph_mlp(x)
        

        return x