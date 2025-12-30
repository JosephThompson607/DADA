
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class EdgeClassifier3Conv3Lin(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super(EdgeClassifier3Conv3Lin, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        edge_input_dim = 2 * hidden_channels + edge_dim

            
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
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
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Edge embedding - gather node features for each edge
        parents, children = edge_index
        edge_data = data.edge_attr
        edge_features = torch.cat([x[parents], x[children], edge_data], dim=1)
        
        # Edge classification
        return self.edge_mlp(edge_features)


class EdgeClassifierMLP(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super(EdgeClassifierMLP, self).__init__()
        edge_input_dim = 2 * hidden_channels + edge_dim

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
        )
            
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data,**data_kwargs):
        x = data.x
        edge_index = data.edge_index
        parents, children = edge_index
        edge_data = data.edge_attr
        x = self.node_mlp(x)
        edge_features = torch.cat([x[parents], x[children], edge_data], dim=1)
        
        return self.edge_mlp(edge_features)


class EdgeClassifierMLP4Layer(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super(EdgeClassifierMLP4Layer, self).__init__()
        edge_input_dim = 2 * hidden_channels + edge_dim

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
        )
            
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data,**data_kwargs):
        x = data.x
        edge_index = data.edge_index
        parents, children = edge_index
        edge_data = data.edge_attr
        x = self.node_mlp(x)
        edge_features = torch.cat([x[parents], x[children], edge_data], dim=1)
        
        return self.edge_mlp(edge_features)
    
class EdgeClassifier(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super(EdgeClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        edge_input_dim = 2 * hidden_channels

            
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim + edge_dim, hidden_channels),
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
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Edge embedding - gather node features for each edge
        parents, children = edge_index
        edge_data = data.edge_attr
        edge_features = torch.cat([x[parents], x[children], edge_data], dim=1)
        
        # Edge classification
        return self.edge_mlp(edge_features)
    
    
class EdgeClassifierGAT(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound using GATv2'''
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, heads=4):
        super(EdgeClassifierGAT, self).__init__()
        
        # 2. GATv2 Configuration
        # Layer 1: Multi-head attention. Output dim becomes hidden_channels * heads
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, 
                               concat=True, edge_dim=edge_dim)
        
        # Layer 2: Project back to hidden_channels. 
        # We set concat=False (averaging) to ensure output is exactly 'hidden_channels'
        # Input dim must match Layer 1 output (hidden_channels * heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, 
                               concat=False, edge_dim=edge_dim)

        edge_input_dim = 2 * hidden_channels+ (edge_dim if edge_dim is not None else 0)



        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data, **data_kwargs):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr # 4. Extract edge attributes early

        #Part 1: GNN foward to node emeddings
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        #Part 2: FC for node to edge embeddings
        parents, children = edge_index
        
        # Concatenate Source Node + Target Node + Edge Features
        if edge_attr is not None:
             edge_features = torch.cat([x[parents], x[children], edge_attr], dim=1)
        else:
             edge_features = torch.cat([x[parents], x[children]], dim=1)
        
        # Edge classification
        return self.edge_mlp(edge_features)

class EdgeClassifierGAT3Layer(torch.nn.Module):
    '''Indicates if an edge has an impact on the SALBP lower bound using GATv2'''
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, heads=4):
        super(EdgeClassifierGAT3Layer, self).__init__()
        
        # 2. GATv2 Configuration
        # Layer 1: Multi-head attention. Output dim becomes hidden_channels * heads
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, 
                               concat=True, edge_dim=edge_dim)
        
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, 
                               concat=True, edge_dim=edge_dim)
        
        # Layer 2: Project back to hidden_channels. 
        # We set concat=False (averaging) to ensure output is exactly 'hidden_channels'
        # Input dim must match Layer 1 output (hidden_channels * heads)
        self.conv3= GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, 
                               concat=False, edge_dim=edge_dim)

        edge_input_dim = 2 * hidden_channels+ (edge_dim if edge_dim is not None else 0)



        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data, **data_kwargs):
        #get node and edge attributes
        x ,  edge_index,edge_attr  = data.x, data.edge_index,data.edge_attr
       
 
        #Part 1: GNN foward to node emeddings
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index,edge_attr= edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index, edge_attr= edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        #Part 2: FC for node to edge embeddings
        parents, children = edge_index
        
        # Concatenate Source Node + Target Node + Edge Features
        if edge_attr is not None:
             edge_features = torch.cat([x[parents], x[children], edge_attr], dim=1)
        else:
             edge_features = torch.cat([x[parents], x[children]], dim=1)
        
        # Edge classification
        return self.edge_mlp(edge_features)
    
# class EdgeClassifierGATStats(torch.nn.Module):
#     '''Indicates if an edge has an impact on the SALBP lower bound'''
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
#         super(EdgeClassifierGATStats, self).__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)  # TODO
#         self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
#                              concat=False, dropout=0.6)  # TODO
#         edge_input_dim = 2 * hidden_channels + 3 #The 3 is from the graph statistics we are calculating
#         self.edge_mlp = torch.nn.Sequential(
#             torch.nn.Linear(edge_input_dim, hidden_channels),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_channels, out_channels)
#         )
        
#     def forward(self, data,**data_kwargs):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         # Apply GAT convolutions to get node embeddings
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         node_embeddings = self.conv2(x, edge_index)
        
#         # Calculate graph statistics from original node features
#         # Assuming x[:, 0] contains the x-values
#         x_values = x[:, 0]
        
#         # If batch information is not provided, assume single graph
#         if batch is None:
#             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
#         stats_tensor = torch.stack(
#             [data_kwargs["graph_data"][fp][instance] for (fp,instance) in zip(data.dataset, data.instance)],
#             dim=0
#         ).to(x.device)
#         # # Calculate per-graph statistics
#         # mean_x = scatter(x_values, batch, dim=0, reduce="mean")
#         # # For std, we need to calculate variance first and then sqrt
#         # var_x = scatter((x_values - mean_x[batch])**2, batch, dim=0, reduce="mean")
#         # std_x = torch.sqrt(var_x + 1e-8)  # Adding small epsilon for numerical stability
#         # min_x = scatter(x_values, batch, dim=0, reduce="min")[0]
#         # max_x = scatter_max(x_values, batch, dim=0, reduce="max")[0]
        
#         # For each edge, get source and target node embeddings
#         row, col = edge_index
#         edge_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        
#         # Add per-graph statistics to each edge feature
#         # Determine which graph each edge belongs to
#         # edge_batch = batch[row]  # Using the source node's batch assignment
        
#         # Create tensor to hold graph stats for each edge
#         # graph_stats = torch.cat([
#         #     mean_x[edge_batch].unsqueeze(1),
#         #     std_x[edge_batch].unsqueeze(1),
#         #     min_x[edge_batch].unsqueeze(1),
#         #     max_x[edge_batch].unsqueeze(1)
#         # ], dim=1)
        
#         # Concatenate edge features with graph statistics
#         edge_features = torch.cat([edge_features, stats_tensor], dim=1)
        
#         # Pass through edge MLP for classification
#         edge_pred = self.edge_mlp(edge_features)
        
#         return edge_pred
    

# class EdgeClassifierGATStats3Layer(torch.nn.Module):
#     '''Indicates if an edge has an impact on the SALBP lower bound'''
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
#         super(EdgeClassifierGATStats, self).__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)  # TODO
#         self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
#                              concat=False, dropout=0.6)  # TODO
#         self.conv3 = GATConv(hidden_channels * heads, hidden_channels,dropout=0.6)  # TODO
#         edge_input_dim = 2 * hidden_channels + 3 #The 3 is from the graph statistics we are calculating
#         self.edge_mlp = torch.nn.Sequential(
#             torch.nn.Linear(edge_input_dim, hidden_channels),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_channels, out_channels)
#         )
        
#     def forward(self, data,**data_kwargs):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         # Apply GAT convolutions to get node embeddings
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         node_embeddings = self.conv2(x, edge_index)
        
#         # Calculate graph statistics from original node features
#         # Assuming x[:, 0] contains the x-values
#         x_values = x[:, 0]
        
#         # If batch information is not provided, assume single graph
#         if batch is None:
#             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
#         stats_tensor = torch.stack(
#             [data_kwargs["graph_data"][fp][instance] for (fp,instance) in zip(data.dataset, data.instance)],
#             dim=0
#         ).to(x.device)
#         # # Calculate per-graph statistics
#         # mean_x = scatter(x_values, batch, dim=0, reduce="mean")
#         # # For std, we need to calculate variance first and then sqrt
#         # var_x = scatter((x_values - mean_x[batch])**2, batch, dim=0, reduce="mean")
#         # std_x = torch.sqrt(var_x + 1e-8)  # Adding small epsilon for numerical stability
#         # min_x = scatter(x_values, batch, dim=0, reduce="min")[0]
#         # max_x = scatter_max(x_values, batch, dim=0, reduce="max")[0]
        
#         # For each edge, get source and target node embeddings
#         row, col = edge_index
#         edge_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        
#         # Add per-graph statistics to each edge feature
#         # Determine which graph each edge belongs to
#         # edge_batch = batch[row]  # Using the source node's batch assignment
        
#         # Create tensor to hold graph stats for each edge
#         # graph_stats = torch.cat([
#         #     mean_x[edge_batch].unsqueeze(1),
#         #     std_x[edge_batch].unsqueeze(1),
#         #     min_x[edge_batch].unsqueeze(1),
#         #     max_x[edge_batch].unsqueeze(1)
#         # ], dim=1)
        
#         # Concatenate edge features with graph statistics
#         edge_features = torch.cat([edge_features, stats_tensor], dim=1)
        
#         # Pass through edge MLP for classification
#         edge_pred = self.edge_mlp(edge_features)
        
#         return edge_pred
    
class GraphClassifier(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, pooling='mean'):
        super(GraphClassifier, self).__init__()
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
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
        x = self.pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_mlp(x)
        

        return x

class GraphClassifier3Layer(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels, out_channels, pooling='mean'):
        super(GraphClassifier3Layer, self).__init__()
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, data,**data_kwargs):
        x ,  edge_index= data.x, data.edge_index
        batch = data.batch
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_mlp(x)
        

        return x
    

class GraphGATClassifier(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels,out_channels,edge_dim=None, heads=4, dropout=0.6):
        super(GraphGATClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads ,edge_dim=edge_dim, dropout=dropout)  
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads, edge_dim=edge_dim,
                             concat=False, dropout=dropout)
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, data, **data_kwargs):
        x ,  edge_index,edge_attr  = data.x, data.edge_index,data.edge_attr
        batch = data.batch
        # Node embedding
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_mlp(x)
        

        return x


class GraphGATClassifier3Layer(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels, hidden_channels,out_channels,edge_dim=None, heads=4, dropout=0.6):
        super(GraphGATClassifier3Layer, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads ,edge_dim=edge_dim, dropout=dropout)  
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads ,edge_dim=edge_dim, dropout=dropout)  
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads, edge_dim=edge_dim,
                             concat=False, dropout=dropout)
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, data, **data_kwargs):
        x ,  edge_index,edge_attr  = data.x, data.edge_index,data.edge_attr
        batch = data.batch
        # Node embedding
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_mlp(x)
        

        return x
    
class GraphClassifierStats(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound, includes graph data after node embeddings'''
    def __init__(self, in_channels,graph_channels,node_indices, graph_indices, hidden_channels, out_channels, pooling='mean'):
        super(GraphClassifierStats, self).__init__()
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        self.register_buffer('node_indices', torch.tensor(node_indices, dtype=torch.long))
        self.register_buffer('graph_indices', torch.tensor(graph_indices, dtype=torch.long))
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels+ graph_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, data,**data_kwargs):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        node_tensor = x[:, self.node_indices] 
        graph_tensor = x[:, self.graph_indices] 
     # Pool graph features (they should be constant per graph, so mean/max/first all work)
        graph_tensor = global_mean_pool(graph_tensor, batch)
        # Node embedding
        x = self.conv1(node_tensor, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        #combine with graph data
        x= torch.cat([x, graph_tensor], dim=1)
        x = self.graph_mlp(x)
        return x

class GraphClassifier3LayerStats(torch.nn.Module):
    '''Indicates if graph has an edge that impacts the lower bound'''
    def __init__(self, in_channels,graph_channels,node_indices, graph_indices, hidden_channels, out_channels, pooling='mean'):
        super(GraphClassifier3LayerStats, self).__init__()
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        self.register_buffer('node_indices', torch.tensor(node_indices, dtype=torch.long))
        self.register_buffer('graph_indices', torch.tensor(graph_indices, dtype=torch.long))
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + graph_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, data,**data_kwargs):
        x ,  edge_index= data.x, data.edge_index
        batch = data.batch
        # feature sorting
        node_tensor = x[:, self.node_indices] 
        graph_tensor = x[:, self.graph_indices] 
        # Pool graph features (they should be constant per graph, so mean/max/first all work)
        graph_tensor = global_mean_pool(graph_tensor, batch)

        #Node embedding
        x = self.conv1(node_tensor, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)

        # Combine with graph features
        x= torch.cat([x, graph_tensor], dim=1)
        x = self.graph_mlp(x)
        return x
    


class GraphGATClassifierStats(torch.nn.Module):
    '''
    GAT-based graph classifier that separates node features (for GNN) 
    and graph-level features (for final MLP).
    '''
    def __init__(self, in_channels, graph_channels, node_indices, graph_indices,
                 hidden_channels, out_channels, edge_dim=None, heads=4, dropout=0.6, pooling='mean'):
        super(GraphGATClassifierStats, self).__init__()
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # Register indices as buffers for proper device management
        self.register_buffer('node_indices', torch.tensor(node_indices, dtype=torch.long))
        self.register_buffer('graph_indices', torch.tensor(graph_indices, dtype=torch.long))
        
        # GAT layers operate on node features only
        self.conv1 = GATConv(in_channels, hidden_channels, heads, 
                            edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads, 
                            edge_dim=edge_dim, concat=False, dropout=dropout)
        
        # Final MLP combines pooled node embeddings with graph features
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + graph_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, data, **data_kwargs):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        
        # Separate node features from graph features
        node_tensor = x[:, self.node_indices]  # [num_nodes, num_node_features]
        graph_features_per_node = x[:, self.graph_indices]  # [num_nodes, num_graph_features]
        graph_tensor = global_mean_pool(graph_features_per_node, batch) #Combine to graph level

        # Node and edge embedding through GAT layers
        x = self.conv1(node_tensor, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        # Pool node embeddings to graph level
        x = self.pool(x, batch)  

        #Combine with Graph data, feed through FC layers
        x = torch.cat([x, graph_tensor], dim=1)  # [batch_size, hidden_channels + graph_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_mlp(x)
        
        return x
    
class GraphGAT3LayerClassifierStats(torch.nn.Module):
    '''
    3-Layer GAT-based graph classifier with 3 FC layers.
    Graph features are seperated out and fed into FC layers at the end.
    '''
    def __init__(self, in_channels, graph_channels, node_indices, graph_indices,
                 hidden_channels, out_channels, edge_dim=None, heads=4, dropout=0.6):
        super(GraphGAT3LayerClassifierStats, self).__init__()
        self.register_buffer('node_indices', torch.tensor(node_indices, dtype=torch.long))
        self.register_buffer('graph_indices', torch.tensor(graph_indices, dtype=torch.long))
        self.dropout= dropout
        #GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads, 
                            edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads, 
                            edge_dim=edge_dim, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads, 
                            edge_dim=edge_dim, concat=False, dropout=dropout)
        
        #FC layers
        self.graph_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + graph_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, data, **data_kwargs):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        
        # Separate node features from graph features
        node_tensor = x[:, self.node_indices]  # [num_nodes, num_node_features]
        graph_features_per_node = x[:, self.graph_indices]  # [num_nodes, num_graph_features]
        graph_tensor = global_mean_pool(graph_features_per_node, batch)  # [batch_size, num_graph_features]
        
        # 3 GAT layers with activation and dropout
        x = self.conv1(node_tensor, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        
        # Pool node embeddings to graph level
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        

        # Concatenate pooled node embeddings with graph features
        x = torch.cat([x, graph_tensor], dim=1)  # [batch_size, hidden_channels + graph_channels]
        
        # 3 FC layers
        x = self.graph_mlp(x)
        
        return x
    

####Multi-Layer Perceptrons###
#These do not use the GCN layers, take features directly 

class GraphRegressorMLP(torch.nn.Module):
    '''Graph-level regression by pooling edge representations'''
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, pooling='mean'):
        super(GraphRegressorMLP, self).__init__()
        
        # Set pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # Node embedding MLP
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
        )
        
        # Edge embedding MLP
        edge_input_dim = 2 * hidden_channels + edge_dim
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Final regression head (after pooling)
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, data, **data_kwargs):
        x = data.x
        edge_index = data.edge_index
        parents, children = edge_index
        edge_data = data.edge_attr
        
        # Get node embeddings
        x = self.node_mlp(x)
        
        # Get edge embeddings
        edge_features = torch.cat([x[parents], x[children], edge_data], dim=1)
        edge_embeddings = self.edge_mlp(edge_features)
        
        # Create edge batch indices (which graph each edge belongs to)
        edge_batch = data.batch[edge_index[0]]
        
        # Pool edge embeddings to graph level
        graph_embedding = self.pool(edge_embeddings, edge_batch)
        
        # Final regression output
        output = self.regression_head(graph_embedding)
        
        return output