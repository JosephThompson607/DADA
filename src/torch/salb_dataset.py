from torch_geometric.data import Dataset, Data, InMemoryDataset
import os.path as osp
from SALBP_solve import *
import torch
import ast
from alb_instance_compressor import open_salbp_pickle_as_dict, open_multi_pickles_as_dict

class SALBDataset(InMemoryDataset):
    def __init__(self, root, edge_data_csv,alb_filepaths=None, transform=None, pre_transform=None,raw_data_folder = "raw/small data set_n=20", cycle_time=1000, from_pickle=True):
        self.raw_data_folder = raw_data_folder
        self.from_pickle = from_pickle
        self.alb_filepaths = alb_filepaths
        
        self.cycle_time = cycle_time
        self.edge_df = pd.read_csv(raw_data_folder + edge_data_csv)
        
        
        if 'alb_files' in self.edge_df.columns:
            self.alb_filepaths = list(self.edge_df['alb_files'].unique())
            
        else:
            self.edge_df['root_fp'] = alb_filepath
            self.edge_df['alb_files']=self.alb_filepath + '/'+ self.edge_df['instance']
        #self.edge_df['alb_files'] = alb_filepath + '/'+ self.edge_df['instance']
        #self.alb_files = self.edge_df['alb_files'].to_list()
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        # Define your custom raw directory
        return osp.normpath( self.raw_data_folder)
    @property
    def processed_dir(self):
        # Define your custom processed directory
        return osp.join(self.root, 'processed/')

    @property
    def raw_file_names(self):
        # List files in `raw_dir` necessary for generating the dataset.
        return self.edge_df['alb_files'].to_list()

    @property
    def processed_file_names(self):
        # List files in `processed_dir` that are already processed.
        pickle_names = self.edge_df['alb_files'].apply(lambda x: x.split('/')[-1].split('.')[0])

        # Create the processed name column
        processed_names = pickle_names  + '-' + self.edge_df['instance'].astype(str) + ".pt"
        processed_names = processed_names.to_list()
        return processed_names
    

    def download(self):
        # Logic for downloading raw data if it does not exist.
        pass

    def process(self):
        # Read raw data and save processed data to `processed_dir`.
        if self.from_pickle:
            salb_instances = open_multi_pickles_as_dict(self.alb_filepaths)
        for index, row in self.edge_df.iterrows():
            print(f"Processing {row['instance']} from file {row['alb_files'].split('/')[-1].split('.')[0]} ...")
            #get name of base pickle file
            pickle_name = row['alb_files'].split('/')[-1].split('.')[0]

            # Create the output filename (must match what processed_file_names returns)
            output_filename = f"{pickle_name}_{row['instance']}.pt"
            output_path = osp.join(self.processed_dir, output_filename)

            # Skip if already processed
            if osp.exists(output_path):
                print(f"Skipping {output_filename}, already processed")
                continue

            if self.from_pickle:
                salb_inst = salb_instances[pickle_name][row['instance']]
            else:
                salb_inst = parse_alb(self.edge_df['root_fp'] + '/' + self.edge_df['instance'] + '.alb')

            prec_relations = [(int(edge[0])-1,int(edge[1]) -1 ) for edge in salb_inst['precedence_relations']]
            edge_index = torch.tensor(prec_relations, dtype=torch.long)
            #loads task times as a value, but keeps it as a fraction of the cycle time
            x = torch.tensor([list(salb_inst['task_times'].values())], dtype=torch.float)/self.cycle_time
               # Properly shape the node features - this is important
            if x.dim() == 2 and x.size(0) == 1:
                # Transpose to get shape [num_nodes, 1]
                x = x.t()
            edge_classes =  ast.literal_eval(row['is_less_max'])
            if len(edge_classes) != len(salb_inst['precedence_relations']):
                print("data mismatch on edges: " ,len(edge_classes), len(salb_inst['precedence_relations']))
                continue
            no_stations = ast.literal_eval(row['no_stations'])

            data = Data(x=x, edge_index=edge_index.t().contiguous())
            data.instance = row['instance']
            data.precendence_relation = ast.literal_eval(row['edge'])
            #print("number of nodes", self.num_nodes)

            data.edge_classes = torch.tensor(edge_classes, dtype =torch.bool)
            data.graph_class = bool(row['min_less_max'])
            data.n_stations = no_stations
            torch.save(data, self.processed_dir + pickle_name + '-' + row['instance'] + ".pt")

    def len(self):
        # Return the number of graphs in the dataset.
        return len(self.processed_file_names)

    def get(self, idx):
        row = self.edge_df.iloc[idx]
        pickle_name = ''
        if  self.from_pickle:
                pickle_name = row['alb_files'].split('/')[-1].split('.')[0]
        # Load and return a graph object by index. NOTE: INDEXING STARTS AT 1 to stay consistent with Otto
        row  = self.edge_df.iloc[idx]
        data = torch.load( self.processed_dir +pickle_name  + '-' +row['instance'] + ".pt", weights_only=False)
        return data

    
    
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
