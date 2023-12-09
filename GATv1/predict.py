import torch
from Dataset.Molecule_dataset import MolecularGraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Dataset.Molecule_dataset import MolecularGraphDataset
from Metrics.metrics import classification_metrics, topk_precision
import torch
from model import GATModel
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
from dotenv import load_dotenv
import os
import numpy as np
import gc
import wandb
from tdc.utils import get_label_map


def label_map_target(labels):  # Obtained from dataset analysis
    # Map labels
    syms = list()
    for labels in labels:
        label_map = get_label_map(
            name='TWOSIDES', task='DDI', name_column='Side Effect Name', path='data/')

        symptoms = [label_map.get(item.item(), item.item()) for item in labels]
        syms.append(symptoms)

    return syms


def predict():  # batch size 1 to get single instance predictions
    for step, graphs in enumerate(test_loader):
        logits, predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        precision, topk_labels = topk_precision(
            predictions, graphs.y.int(), k=5)

    top_symptoms = label_map_target(topk_labels)

    return precision, top_symptoms


if __name__ == '__main__':

    load_dotenv(".env")
    test_folds = ['fold8']
    test_set = MolecularGraphDataset(fold_key=test_folds[0], root=os.getenv(
        "graph_files")+"/fold8"+"/data/", start=52500)

    params = {
        "batch_size": 2,
        'shuffle': True
    }

    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    model = GATModel(dataset=test_set)  # For tensor dimensions

    model.load_state_dict(torch.load(
        "GAT/weights/train_fold_12/head_1/model70.pth"))

    # Get the Predictions with Scores
    prec, symps = predict()
    print("Confidence :", prec)
    print("Symptoms: ", symps)
