from torch_geometric.data import Dataset, Data, InMemoryDataset
import os.path as osp
from SALBP_solve import *
import torch
import ast
from alb_instance_compressor import open_salbp_pickle_as_dict, open_multi_pickles_as_dict
from tqdm import tqdm
import pandas as pd
from typing import List, Optional


def get_x_feature_vector(salbp_inst, instance_df, debug_time = False):
        df = instance_df[[ "edge",  "avg_div_c", "child_in_degree", "random_max_gap", 
                                        "nodes_in_chains", "min_div_c", "neighborhood_std", 
                                        "random_avg_gap", "priority_min_gap", "random_min_gap", "priority_max_gap", 
                                         "prec_bias", "random_coefficient_of_variation", 
                                        "max_div_c", "divergence_degree", 
                                         "average_number_of_immediate_predecessors", 
                                        "order_strength", "std_div_c",  "convergence_degree", "ti_size", 
                                         "child_rw_mean", 
                                        "child_rw_max",
                                        "load_child_mean", 
                                        "child_pos_weight",
                                        "child_rw_mean_total_time",
                                        "child_rw_mean_max_time", 
                                        "child_weight", 
                                        "parent_weight", 
                                        "parent_pos_weight",
                                        "load_parent_mean", 
                                        "parent_out_degree",
                                        "rw_mean_total_time", 
                                        "rw_max", 
                                        "rw_mean",
                                        "rw_mean_max_time" ]].copy()

  

        # Step 1: Extract parent and child IDs from edge column
        df["parent"] = df["edge"].apply(lambda e: str(e[0]) if isinstance(e, (list, tuple)) else str(eval(e)[0]))
        df["child"] = df["edge"].apply(lambda e: str(e[1]) if isinstance(e, (list, tuple)) else str(eval(e)[1]))

        # Step 2: Separate feature columns
        feature_cols = [c for c in df.columns if c not in ["edge", "parent", "child"]]

        # Step 3: Identify which columns belong to parent/child/both
        parent_cols = [c for c in feature_cols if ("parent" in c or (c.startswith("rw") and "child" not in c))]
        child_cols = [c for c in feature_cols if "child" in c]
        both_cols = [c for c in feature_cols if c not in parent_cols + child_cols]

        # Step 4: Build parent rows
        parent_df = df[["parent"] + parent_cols + both_cols].copy()
        parent_df = parent_df.rename(columns={"parent": "node"})
        parent_df.columns = [c.replace("parent_", "") for c in parent_df.columns]

        # Step 5: Build child rows
        child_df = df[["child"] + child_cols + both_cols].copy()
        child_df = child_df.rename(columns={"child": "node"})
        child_df.columns = [c.replace("child_", "") for c in child_df.columns]

        # Step 6: Combine both and aggregate duplicates
        node_df = pd.concat([parent_df, child_df], ignore_index=True)
        
        # --- Check for duplicates with differing values ---
        if debug_time:
            dupes = node_df[node_df.duplicated("node", keep=False)].sort_values("node")
            if not dupes.empty:
                diff_nodes = []
                for n, g in dupes.groupby("node"):
                    if not g.drop(columns="node").nunique().eq(1).all():
                        diff_nodes.append(n)
                        print("dupe detected: ", g)
                if diff_nodes:
                    print(f"⚠️ Warning: nodes with inconsistent duplicate values: {diff_nodes}")

        
        nan_rows = node_df[node_df.isna().any(axis=1)]
        if debug_time:
            if not nan_rows.empty:
                print("⚠️ Warning: rows with NaN values detected:")
                print(nan_rows)
        #Treat data for NN use
        # If nodes appear multiple times, aggregate (mean or other)
        node_df = node_df.groupby("node").mean().reset_index().fillna(0)
        task_df = pd.DataFrame(list(salbp_inst['task_times'].items()), columns=["node", "value"])
        task_df["node"] = task_df["node"].astype(int)
        node_df["node"] = node_df["node"].astype(int)
        task_df = task_df.merge(node_df, on="node", how= "left").fillna(0)
        if debug_time:
            nan_rows = task_df[task_df.isna().any(axis=1)]
            if not nan_rows.empty:
                print("⚠️ Warning: rows with NaN values detected for the joined dataframe:")
                print(nan_rows)
        
        # Sort rows by node ID
        task_df = task_df.sort_values("node").reset_index(drop=True)
        assert task_df["node"].is_monotonic_increasing, "Nodes not sorted properly!"
        assert task_df["node"].nunique() == len(salbp_inst["task_times"].keys()), "Node mismatch!"
        assert task_df["node"].min() == 0 or task_df["node"].min() == 1, "Unexpected node numbering!"
        # Extract feature matrix (drop the node ID column)
        x_values = task_df.drop(columns=["node"]).values

        # Convert to a PyTorch tensor (float type)
        x = torch.tensor(x_values, dtype=torch.float)
        
        return x
        


class EdgeClassificationDataset(InMemoryDataset):
    def __init__(self, csv_path, pickle_dir, root):
        self.csv_path = csv_path
        self.pickle_dir = pickle_dir
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.csv_path 
    
    @property
    def processed_file_names(self):
        # One file per instance
        # df = pd.read_csv(self.csv_path)
        # return [f'data_{i}.pt' for i in range(len(df.groupby('instance')))]
        #One file for all instances
        return ['processed.pt']
    
    def process(self):
        data_list = []
        df = pd.read_csv(self.csv_path)
        

        if 'no_stations' in df.columns:
            df.rename(columns={'no_stations':'n_stations'}, inplace=True)
        # Load pickles
        graphs = {}
        for pkl_file in df['pickle_file'].unique():
            with open(f"{self.pickle_dir}/{pkl_file}", 'rb') as f:
                pickle_list =  pickle.load(f)
                pkl_dict = {}
                for instance in pickle_list:
                    name = str(instance['name']).split('/')[-1].split('.')[0]
                    pkl_dict[name] = instance
                graphs[pkl_file] = pkl_dict
        # Process each instance
        for idx, (instance_name, instance_df) in enumerate(df.groupby('instance')):
            pkl_file = instance_df['pickle_file'].iloc[0]
            salb_inst = graphs[pkl_file][instance_name]

            # Extract dataset name from pickle filename
            dataset_name = pkl_file.split('/')[-1].replace('.pkl', '')
            #Node level features will be everything that is attributed to a node as well as graph level data
            

            x = get_x_feature_vector(salb_inst, instance_df)
            edge_features = instance_df[[
                                            "chain_avg",
                                            "chain_max",
                                            "chain_std",
                                            "chain_min",
                                        ]].values
            edge_labels = (instance_df['s_orig'] > instance_df['n_stations']).astype(int).values
            graph_label = int(edge_labels.any())  # 1 if any edge label is True, else 0
            
            #Getting graph data
            prec_relations = [(int(edge[0]) - 1, int(edge[1]) - 1) for edge in salb_inst['precedence_relations']]
            edge_index = torch.tensor(prec_relations, dtype=torch.long).t().contiguous()
            
            #creating data object
            data = Data(
                x=x,
                y=torch.tensor([graph_label], dtype=torch.long),  # graph label

                edge_index=edge_index,
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                edge_label=torch.tensor(edge_labels, dtype=torch.long),
                instance_name=instance_name,
                dataset_name=dataset_name,
                edge_names=instance_df['edge'].tolist() 

            )
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    




def create_and_load_salbp_dataset(root_fp, pkl_fp, dataset_name):
    with open(pkl_fp, 'rb') as f:
        data = pickle.load(f)    

    #assuming datset has the name {name}_geo_ready.pkl
    salbp_ds = SALBPGNNDataset(root_fp, data,  dataset_name,)
    return salbp_ds


class SALBPGNNDataset(InMemoryDataset):
    """
    InMemoryDataset that stores all node and edge features with support for both 
    node-level and graph-level tasks. Allows flexible feature slicing by name.
    
    Features:
    - Stores all features in memory for fast access
    - Supports node features, edge features, edge labels, and graph labels
    - Feature slicing via select_features() without reprocessing
    - Tracks feature column names for interpretability
    - Expects all numeric data as torch.Tensors (no conversion performed)
    
    Args:
        root: Root directory for storing processed dataset
        graphs: List of graph dictionaries (see structure below)
        dataset_name: Unique identifier for this dataset (used in filename)
        transform: Optional transform applied when accessing data
        pre_transform: Optional transform applied during processing
    
    Expected graph structure (dict keys):
        Required:
            'instance': str - Instance name/identifier
            'edges': List[List[int]] - Edge indices in format [[sources], [targets]], shape [2, num_edges]
            'features': torch.Tensor - Node features, shape [num_nodes, num_features]
        
        Optional - Node features:
            'x_cols': List[str] - Names of node feature columns (for tracking)
        
        Optional - Edge features:
            'edge_features': torch.Tensor - Edge attributes, shape [num_edges, num_edge_features]
            'edge_level_features': List[str] - Names of edge feature columns
        
        Optional - Edge labels (for edge-level prediction):
            'edge_label_values': torch.Tensor - Edge label values, shape [num_edges] or [num_edges, num_labels]
            'edge_labels': List[str] - Names of edge label columns
        
        Optional - Graph labels (for graph-level prediction):
            'graph_label_values': torch.Tensor - Graph-level label(s), shape [num_labels] or scalar
            'graph_labels': List[str] - Names of graph label columns
    
    Data object attributes after processing:
        - instance_name: str - Instance identifier
        - x: torch.Tensor [num_nodes, num_features] - Node features
        - x_cols: List[str] - Node feature column names
        - edge_index: torch.Tensor [2, num_edges] - Edge connectivity
        - edge_attr: torch.Tensor [num_edges, num_edge_features] - Edge features
        - edge_cols: List[str] - Edge feature column names
        - y: torch.Tensor - Graph-level labels (if provided)
        - graph_labels: List[str] - Graph label column names
        - y_edge: torch.Tensor - Edge-level labels (if provided)
        - edge_labels: List[str] - Edge label column names 
    
    Usage:
        # Create dataset with all features
        dataset = SALBPGNNDataset(
            root='./data',
            graphs=graph_list,
            dataset_name='my_dataset'
        )
        
        # Access full dataset
        data = dataset[0]  # Gets first graph with all features
        
        # Create view with selected features only
        subset = dataset.select_features(['feat1', 'feat3'])
        data = subset[0]  # Gets first graph with only feat1 and feat3
    
    Notes:
        - All input numeric data (features, labels) must already be torch.Tensors
        - Only 'edges' list is converted to tensor during processing
        - All features are stored during process() - feature selection is done at access time
        - Processed data is cached to disk and reloaded on subsequent runs
        - Both node-level and graph-level tasks are supported simultaneously
        - Edge features and labels are independent of node feature slicing

    """
    def __init__(
        self,
        root: str,
        graphs: List,
        dataset_name: str,
        transform=None,
        pre_transform=None
    ):
        self.graphs = graphs
        self.dataset_name = dataset_name
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}_all_features.pt']
    
    def download(self):
        pass
    
    def process(self):
        """Process graphs and store ALL features."""
        data_list = []
        
        for graph_data in self.graphs:
            # Edges
            instance_name = graph_data['instance']
            edge_index = torch.tensor(graph_data['edges'], dtype=torch.long)
            
            # Store ALL features
            x = graph_data['features']
            if 'x_cols' in graph_data:
                x_cols = graph_data['x_cols']
            else:
                x_cols =None
                print('Warning: Node feature columns are unspecified')
            
            #Edge labels
            y_edge = None
            edge_labels = None
            if 'edge_label_values' in graph_data and graph_data['edge_label_values'] is not None:
                y_edge = graph_data['edge_label_values'] 
                if 'edge_labels' in graph_data:
                    edge_labels = graph_data['edge_labels']
                else: 
                    print('Warning, edge labels are unspecified ')
            #graph-level label
            y = None
            graph_labels = None
            if 'graph_label_values' in graph_data and graph_data['graph_label_values'] is not None:
                y = graph_data['graph_label_values']
                if 'graph_labels' in graph_data:
                    graph_labels = graph_data['graph_labels']
                else:
                    print("Warning, graph labels are unspecified")
                
            if  y is None and y_edge is None:
                print("Warning: no labels in the dataset")
            # Optional: edge attributes
            edge_attr = None
            if 'edge_features' in graph_data:
                edge_attr = graph_data['edge_features']
            
            # Optional: edge weights
            edge_cols = None
            if 'edge_level_features' in graph_data:
                edge_cols = graph_data['edge_level_features']
            data = Data(
                instance_name = instance_name,
                x=x,
                x_cols = x_cols,
                edge_index=edge_index,
                y=y,
                graph_labels = graph_labels,
                y_edge = y_edge,
                edge_labels = edge_labels,
                edge_attr=edge_attr,
                edge_cols=edge_cols
            )
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def select_features(self, selected_features: List[str]):
        """
        Return a new dataset view with only selected features.
        
        Args:
            selected_features: List of feature names to include
            
        Returns:
            New dataset instance with sliced features
        """
        # Get feature indices
        
        
        # Create sliced version
        sliced_dataset = FeatureSlicedDataset(
            self, 
            selected_features
        )
        
        return sliced_dataset
        
class FeatureSlicedDataset:
    """
    A view of a SALBPGNNDataset with sliced features.
    Behaves like a PyG dataset but doesn't store duplicate data.
    """
    
    def __init__(
        self, 
        parent_dataset: SALBPGNNDataset, 
        selected_feature_names: List[str]
    ):
        self.parent = parent_dataset
        self.selected_feature_names = selected_feature_names
        #Get available features
        data_0 = self.parent[0].clone()
        self.feature_names = data_0.x_cols
        #Get the indices of the selected features
        self.selected_indices = [self.feature_names.index(name) for name in self.selected_feature_names]
    def __len__(self):
        return len(self.parent)
    
    def __getitem__(self, idx):
        """Get graph with sliced features."""
        
        data = self.parent[idx].clone()
        data.x = data.x[:, self.selected_indices]
        return data
    
    def __repr__(self):
        return (f'{self.__class__.__name__}({len(self)}, '
                f'features={self.selected_feature_names})')