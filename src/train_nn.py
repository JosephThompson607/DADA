import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import graphviz
import torch_geometric
from torch_geometric.data import Dataset, Data, InMemoryDataset

from torch_geometric.loader import DataLoader
import os.path as osp
import ast
from sklearn.preprocessing import MinMaxScaler
import torchmetrics
import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath("src/torch"))
from salb_dataset import *
from gnns import *
from SALBP_solve import *
from torch.utils.tensorboard import SummaryWriter
import yaml



def train_edge_classifier(input_dataset, config ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:", device)
    best_loss = 10000

    # Hyperparameters
   
    #splits the data into train and test
    train_dataset, test_dataset = random_split(input_dataset, [int(len(input_dataset)*0.8), len(input_dataset) - int(len(input_dataset)*0.8)])
    # train_dataset = [input_dataset[3]]
    # test_dataset = [input_dataset[3]]
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=config["workers"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=config["workers"])

    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    f1_score = torchmetrics.F1Score(task="binary").to(device)


    # Get today's date as a string
    today_str = datetime.today().strftime('%Y-%m-%d')

    # Use it in a filename
    xp_name = config["xp_name"]
    filename = f"{xp_name}_{today_str}"
    writer = SummaryWriter(f"runs/{filename}")

    model = config["nn"]["architecture"](**config["nn"]["params"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # or 'max' depending on the metric
        factor=config['lr_scheduler']['factor'],          # multiply LR by this factor
        patience=config['lr_scheduler']['patience'],         # wait N epochs with no improvement
        min_lr = config['lr_scheduler']['min_lr'],

    )
    pos_weight = get_pos_weight(input_dataset)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    #loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    model.train()
    for epoch in range(config["epochs"]):
        total_loss = 0
        # total_correct = 0
        # total = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out.squeeze(1), data.edge_classes.float())
            # probs = torch.sigmoid(out)
            # preds = (probs > 0.5).int().squeeze(1)
            # total_correct += (preds == data.edge_classes)
            # total += len(preds)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()



        writer.add_scalar('Loss/train', total_loss, epoch)
        # acc = toal_correct/total
        #writer.add_scalar('Accuracy/train', acc, epoch)
        # writer.add_scalar('Recall/train', rec, epoch)
        # writer.add_scalar('Precision/train', prec, epoch)

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

            
    
            # Evaluation
        model.eval()
        #  Testing loop
        with torch.no_grad():
            test_total_loss = 0
            acc = 0
            prec =0
            rec = 0
            f1 = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                test_loss = loss_fn(out.squeeze(1), data.edge_classes.float())
                test_total_loss += test_loss.item()
                probs = torch.sigmoid(out)
                # Set a threshold to convert probabilities to binary predictions
                threshold = 0.5
                targets = data.edge_classes
                preds = (probs > threshold).int().squeeze(1)
                acc += accuracy(preds, targets)/len(test_loader)
                prec += precision(preds, targets)/len(test_loader)
                rec += recall(preds, targets)/len(test_loader)
                f1 += f1_score(preds, targets)/len(test_loader)
            writer.add_scalar('Accuracy/test', acc, epoch)
            writer.add_scalar('Recall/test', rec, epoch)
            writer.add_scalar('Precision/test', prec, epoch)
            writer.add_scalar('F1/test', f1, epoch, epoch)
            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f} average probability {probs.squeeze(1).float().mean()}")
            print("test_total_loss:",test_total_loss, "best lost: ", best_loss)
            if test_total_loss < best_loss:
                    print("test_total_loss:",test_total_loss, "best lost: ", best_loss, "saving trained weights")
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f'checkpoints/{filename}_checkpoint.pth')
                    best_loss = test_total_loss
            scheduler.step(test_total_loss)

def train_graph_classifier(input_dataset, config ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:", device)
    best_loss = 10000

    # Hyperparameters
   
    #splits the data into train and test
    train_dataset, test_dataset = random_split(input_dataset, [int(len(input_dataset)*0.8), len(input_dataset) - int(len(input_dataset)*0.8)])
    # train_dataset = [input_dataset[3]]
    # test_dataset = [input_dataset[3]]
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=config["workers"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=config["workers"])

    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    f1_score = torchmetrics.F1Score(task="binary").to(device)


    # Get today's date as a string
    today_str = datetime.today().strftime('%Y-%m-%d')

    # Use it in a filename
    xp_name = config["xp_name"]
    filename = f"{xp_name}_{today_str}"
    writer = SummaryWriter(f"runs/{filename}")

    model = config["nn"]["architecture"](**config["nn"]["params"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # or 'max' depending on the metric
        factor=config['lr_scheduler']['factor'],          # multiply LR by this factor
        patience=config['lr_scheduler']['patience'],         # wait N epochs with no improvement
        min_lr = config['lr_scheduler']['min_lr'],

    )
    pos_weight = get_pos_weight_graph(input_dataset)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    #loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    model.train()
    for epoch in range(config["epochs"]):
        total_loss = 0
        # total_correct = 0
        # total = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out.squeeze(1), data.graph_class.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()



        writer.add_scalar('Loss/train', total_loss, epoch)
        # acc = toal_correct/total
        #writer.add_scalar('Accuracy/train', acc, epoch)
        # writer.add_scalar('Recall/train', rec, epoch)
        # writer.add_scalar('Precision/train', prec, epoch)

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

            
    
            # Evaluation
        model.eval()
        #  Testing loop
        with torch.no_grad():
            test_total_loss = 0
            acc = 0
            prec =0
            rec = 0
            f1 = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                test_loss = loss_fn(out.squeeze(1), data.graph_class.float())
                test_total_loss += test_loss.item()
                probs = torch.sigmoid(out)
                # Set a threshold to convert probabilities to binary predictions
                threshold = 0.5
                targets = data.graph_class
                preds = (probs > threshold).int().squeeze(1)
                acc += accuracy(preds, targets)/len(test_loader)
                prec += precision(preds, targets)/len(test_loader)
                rec += recall(preds, targets)/len(test_loader)
                f1 += f1_score(preds, targets)/len(test_loader)
            writer.add_scalar('Accuracy/test', acc, epoch)
            writer.add_scalar('Recall/test', rec, epoch)
            writer.add_scalar('Precision/test', prec, epoch)
            writer.add_scalar('F1/test', f1, epoch, epoch)
            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f} average probability {probs.squeeze(1).float().mean()}")
            print("test_total_loss:",test_total_loss, "best lost: ", best_loss, " last lr: ", scheduler.get_last_lr())
            if test_total_loss < best_loss:
                    print("test_total_loss:",test_total_loss, "best lost: ", best_loss, "saving trained weights")
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f'checkpoints/{filename}_checkpoint.pth')
                    best_loss = test_total_loss
            scheduler.step(test_total_loss)



# If architecture refers to a class, pass a mapping like {'EdgeClassifier_GAT': EdgeClassifier_GAT}
def load_config(config_path, architecture_map=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Optionally convert architecture string to actual class
    print(config['nn'], 'architecture' in config['nn'])
    if architecture_map and 'architecture' in config['nn']:
        arch_name = config['nn']['architecture']
        if arch_name in architecture_map:
            config['nn']['architecture'] = architecture_map[arch_name]
            print("using: ", arch_name, " nn configuration")
        else:
            raise ValueError(f"Unknown architecture: {arch_name}")

    return config
def main():
    architecture_map = {
    "EdgeClassifier_GAT": EdgeClassifier_GAT,
    "EdgeClassifier_GNN": EdgeClassifier,
    "EdgeClassifierGATStats": EdgeClassifierGATStats,
    "GraphClassifier": GraphClassifier,
}
    # Create argument parser
    parser = argparse.ArgumentParser(description='Solve edge removal on SALBP instance')
    
    # Add arguments
    # parser.add_argument('--start', type=int, required=False, help='Starting integer (inclusive)')
    # parser.add_argument('--end', type=int, required=False, help='Ending integer (inclusive)')
    # parser.add_argument('--n_processes', type=int, required=False, default=1, help='Number of processes to use')
    # parser.add_argument('--from_alb_folder', action="store_true", help='Whether to read albs directly from a folder, if false, reads from pickle')
    parser.add_argument('--config_yaml', type=str, required=True, help='Configuration file of the expirement')
    #parser.add_argument('--alb_fp', type=str, required=False, help='filepath of alb files if not provided in the csv of the raw instances')
    # parser.add_argument('--n_workers', type=int, required=False, default=1, help='Number of workers for dataloader to use')
    #parser.add_argument('--epochs',type=int, required=False, default=2000, help='Number Epochs for training')
    # parser.add_argument('--n_hid_layers',type=int, required=False, default=64, help='Number of hidden layers to use')
    # parser.add_argument('--att_heads',type=int, required=False, default=4, help='Number of attention heads to use')
    # parser.add_argument('--ds_root', type=str, required=False, help='Neural network architecture to use. Default is EdgeClassifier_GAT')
    # parser.add_argument('--batch_size',type=int, required=False, default=64, help='Batch size to use')
    # parser.add_argument('--backup_name', type=str, required=True, help='name for intermediate saves')
    # parser.add_argument('--filepath', type=str, required=True, help='filepath for alb dataset')
    # parser.add_argument('--instance_name', type=str, required=False, help='start of instance name EX: "instance_n=50_"')
    # parser.add_argument('--final_results_fp', type=str, required=True, help='filepath for results, if no error')
    
    # Parse arguments
    args = parser.parse_args()
    config = load_config(args.config_yaml, architecture_map=architecture_map)
    print("loading the dataset")
    if 'pickle_loc' in config.keys():
        my_dataset = SALBDataset(alb_filepaths=config["pickle_loc"], root=config["data"]["ds_root"], edge_data_csv =config["data"]["instance_csv"],  raw_data_folder =config["data"]["raw_data_dir"])
    else:
        my_dataset = SALBDataset(root=config["data"]["ds_root"], edge_data_csv =config["data"]["instance_csv"],  raw_data_folder =config["data"]["raw_data_dir"])


   # print("filtering for positive instances (edge classification)")
    #my_dataset =  [data for data in my_dataset if data.graph_class] inefficient TODO: find better way, currently using preprocessed dataset
    print("done loading dataset")
    print("config xp type: ", config['xp_type'] )
    if config['xp_type'] == 'edge_classification':
        train_edge_classifier(my_dataset, config )
    elif config['xp_type'] == 'graph_classification':
        train_graph_classifier(my_dataset, config)
    else:
        print("Error in xp type, must choose either edge_classification or graph_classification")
    # my_dataset = SALBDataset(root='pytorch_datasets/n_50',edge_data_csv ="n_50_all.csv",  raw_data_folder ="pytorch_datasets/n_50/raw/")
    #my_dataset = [data for data in my_dataset if data.graph_class]


if __name__ == "__main__":
    main()