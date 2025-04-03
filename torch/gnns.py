
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch.utils.data import random_split

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
        
    def forward(self, x, edge_index):
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
    
class GraphClassifier(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels,  num_classes=1):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, batch ):
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        

        return x