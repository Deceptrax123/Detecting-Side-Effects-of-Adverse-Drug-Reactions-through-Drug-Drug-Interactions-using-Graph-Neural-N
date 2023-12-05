from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from Dataset.Molecule_dataset import MolecularGraphDataset
import torch
from model import GATModel
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from dotenv import load_dotenv
import os
import gc
import wandb

if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    load_dotenv(".env")

    # Set the training and testing folds
    train_folds = ['fold1', 'fold2']
    test_fold = 'fold3'

    params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 0
    }

    train_set1 = MolecularGraphDataset(
        fold_key=train_folds[0], root=os.getenv("graph_files")+"/fold1/"+"/data/", start=0)
    train_set2 = MolecularGraphDataset(fold_key=train_folds[1], root=os.getenv("graph_files")+"/fold2/"
                                       + "/data/", start=7500)

    test_set = MolecularGraphDataset(test_fold, root=os.getenv(
        "graph_files")+"/fold3/"+"/data/", start=15000)
    train_set = ConcatDataset([train_set1, train_set2])

    train_loader = DataLoader(train_set, **params, follow_batch=['x_s', 'x_t'])
    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])
