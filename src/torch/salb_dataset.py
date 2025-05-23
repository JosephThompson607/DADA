from torch_geometric.data import Dataset, Data, InMemoryDataset
import os.path as osp
from SALBP_solve import *
import torch
import ast
from alb_instance_compressor import open_salbp_pickle_as_dict, open_multi_pickles_as_dict
from tqdm import tqdm
import pandas as pd

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
