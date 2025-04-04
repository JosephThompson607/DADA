from torch_geometric.data import Dataset, Data, InMemoryDataset
import os.path as osp
from SALBP_solve import *
import torch
import ast

class SALBDataset(InMemoryDataset):
    def __init__(self, root, edge_data_csv,alb_filepath, transform=None, pre_transform=None,raw_data_folder = "raw/small data set_n=20", cycle_time=1000):
        self.raw_data_folder = raw_data_folder
        self.alb_filepath = alb_filepath
        self.cycle_time = cycle_time
        self.edge_df = pd.read_csv(raw_data_folder + edge_data_csv)
        self.edge_df['alb_files'] = alb_filepath + '/'+ self.edge_df['instance']
        self.alb_files = self.edge_df['alb_files'].to_list()
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
        return self.alb_files

    @property
    def processed_file_names(self):
        # List files in `processed_dir` that are already processed.
        processed_names = self.edge_df['instance'] + ".pt" 
        return processed_names.to_list()
    

    def download(self):
        # Logic for downloading raw data if it does not exist.
        pass

    def process(self):
        # Read raw data and save processed data to `processed_dir`.
        for index, row in self.edge_df.iterrows():
            print(f"Processing {row['instance']}...")
            # Example: Read graph data
            print("processing", row['alb_files'])
            salb_inst = parse_alb(row['alb_files'] + '.alb')
            edge_classes =  ast.literal_eval(row['is_less_max'])
            if len(edge_classes) != len(salb_inst['precedence_relations']):
                print("data mismatch on edges: " ,len(edge_classes), len(salb_inst['precedence_relations']))
                continue
            no_stations = ast.literal_eval(row['no_stations'])
            prec_relations = [(int(edge[0])-1,int(edge[1]) -1 ) for edge in salb_inst['precedence_relations']]
            edge_index = torch.tensor(prec_relations, dtype=torch.long)
            #loads task times as a value, but keeps it as a fraction of the cycle time
            x = torch.tensor([list(salb_inst['task_times'].values())], dtype=torch.float)/self.cycle_time
               # Properly shape the node features - this is important
            if x.dim() == 2 and x.size(0) == 1:
                # Transpose to get shape [num_nodes, 1]
                x = x.t()
            data = Data(x=x, edge_index=edge_index.t().contiguous())
            data.instance = row['instance']
            data.precendence_relation = ast.literal_eval(row['edge'])
            #print("number of nodes", self.num_nodes)
            
            data.edge_classes = torch.tensor(edge_classes, dtype =torch.bool)
            data.graph_class = bool(row['min_less_max'])
            data.n_stations = no_stations
            torch.save(data, self.processed_dir +row['instance'] + ".pt")

    def len(self):
        # Return the number of graphs in the dataset.
        return len(self.processed_file_names)

    def get(self, idx):
        # Load and return a graph object by index. NOTE: INDEXING STARTS AT 1 to stay consistent with Otto
        row  = self.edge_df.iloc[idx]
        data = torch.load( self.processed_dir +row['instance'] + ".pt", weights_only=False)
        return data
