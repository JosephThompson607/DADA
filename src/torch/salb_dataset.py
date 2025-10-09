from torch_geometric.data import Dataset, Data, InMemoryDataset
import os.path as osp
from SALBP_solve import *
import torch
import ast
from alb_instance_compressor import open_salbp_pickle_as_dict, open_multi_pickles_as_dict
from tqdm import tqdm
import pandas as pd


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
        task_df = task_df.merge(node_df, on="node", how= "left")
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
        


class EdgeClassificationDataset(Dataset):
    def __init__(self, csv_path, pickle_dir, root):
        self.csv_path = csv_path
        self.pickle_dir = pickle_dir
        super().__init__(root)
        
    @property
    def raw_file_names(self):
        return self.csv_path 
    
    @property
    def processed_file_names(self):
        # One file per instance
        df = pd.read_csv(self.csv_path)
        return [f'data_{i}.pt' for i in range(len(df.groupby('instance')))]
    
    def process(self):
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

            torch.save(data, self.processed_paths[idx])
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        return torch.load(self.processed_paths[idx])

class SALBDataset(InMemoryDataset):
    def __init__(self, root, edge_data_csv, alb_filepaths=None, transform=None, pre_transform=None, 
                 raw_data_folder="raw/small data set_n=20", cycle_time=1000, from_pickle=True):
        
        self.raw_data_folder = raw_data_folder
        self.from_pickle = from_pickle
        self.alb_filepaths = alb_filepaths
        self.cycle_time = cycle_time

        self.edge_df = pd.read_csv(osp.join(raw_data_folder, edge_data_csv))

        if 'alb_files' in self.edge_df.columns:
            self.alb_filepaths = list(self.edge_df['alb_files'].unique())
        else:
            self.alb_filepaths = list(alb_filepaths)

        super().__init__(root, transform, pre_transform)

        # Load all data from the single processed .pt file
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return osp.normpath(self.raw_data_folder)

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed/')

    @property
    def raw_file_names(self):
        return self.edge_df['alb_files'].tolist()

    @property
    def processed_file_names(self):
        return ["salb_dataset.pt"]

    def download(self):
        pass  # Not used here

    def process(self):
        print("Processing all data and storing in memory...")
        data_list = []

        if self.from_pickle:
            salb_instances = open_multi_pickles_as_dict(self.alb_filepaths)

        for idx, row in tqdm(self.edge_df.iterrows(), total=len(self.edge_df)):
            pickle_name = row['alb_files'].split('/')[-1].split('.')[0]
            instance_name = row['instance']

            if self.from_pickle:
                salb_inst = salb_instances[pickle_name][instance_name]
            else:
                salb_inst = parse_alb(row['root_fp'] + '/' + instance_name + '.alb')

            prec_relations = [(int(edge[0]) - 1, int(edge[1]) - 1) for edge in salb_inst['precedence_relations']]
            edge_index = torch.tensor(prec_relations, dtype=torch.long).t().contiguous()

            x = torch.tensor([list(salb_inst['task_times'].values())], dtype=torch.float) / self.cycle_time
            if x.dim() == 2 and x.size(0) == 1:
                x = x.t()

            edge_classes = ast.literal_eval(row['is_less_max'])
            no_stations = ast.literal_eval(row['no_stations'])

            if len(edge_classes) != len(salb_inst['precedence_relations']):
                print(f"Skipping {instance_name}: mismatch in edge lengths")
                continue
            if 'dataset' not in row.keys():
                row['dataset'] = None
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_classes=torch.tensor(edge_classes, dtype=torch.bool),
                graph_class=bool(row['min_less_max']),
                n_stations=no_stations,
                instance=instance_name,
                precedence_relation=ast.literal_eval(row['edge']),
                dataset = row['alb_files']
            )

            data_list.append(data)

        # Final save in PyG format
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])    
    
def get_pos_weight(data_list):
    
    num_pos = sum(data.edge_classes.sum().item() for data in data_list)
    num_neg = sum(len(data.edge_classes) - data.edge_classes.sum().item() for data in data_list)
    
    # Compute pos_weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([num_neg / num_pos])  # Shape: (1,)
    print(f"num pos {num_pos} num neg {num_neg} pos weight {pos_weight}")
    percent_pos = num_pos/(num_pos + num_neg)
    percent_neg = num_neg/(num_pos + num_neg)
    print(f"Percent positive {percent_pos} percent negative {percent_neg}")
    return pos_weight


def get_pos_weight_graph(data_list):
    num_pos = sum(1 for data in data_list if data.graph_class)
    num_neg = len(data_list) - num_pos
    
    # Compute pos_weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([num_neg / num_pos])  # Shape: (1,)
    print(f"num pos {num_pos} num neg {num_neg} pos weight {pos_weight}")
    percent_pos = num_pos/(num_pos + num_neg)
    percent_neg = num_neg/(num_pos + num_neg)
    print(f"Percent positive {percent_pos} percent negative {percent_neg}")
    return pos_weight
