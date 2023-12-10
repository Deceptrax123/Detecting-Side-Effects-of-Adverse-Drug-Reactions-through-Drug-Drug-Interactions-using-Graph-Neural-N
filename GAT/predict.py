import torch
from Dataset.Molecule_dataset import MolecularGraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Dataset.test_molecule_dataset import TestMolecularGraphDataset
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
    for l in labels:
        label_map = get_label_map(
            name='TWOSIDES', task='DDI', name_column='Side Effect Name', path='data/')

        symptoms = [label_map.get(item.item(), item.item()) for item in l]
        syms.append(symptoms)

    return syms


def predict():  # batch size 1 to get single instance predictions

    precisions = list()
    labels = list()
    scores = list()
    for step, graphs in enumerate(test_loader):
        logits, predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        precision, topk_labels, score = topk_precision(
            predictions, graphs.y.int(), k=3)

        top_symptoms = label_map_target(topk_labels)

        precisions.append(precision)
        labels.append(top_symptoms)
        scores.append(score)

    return sum(precisions)/len(precisions), labels, sum(scores)/len(scores)


if __name__ == '__main__':

    load_dotenv(".env")
    test_set = TestMolecularGraphDataset(fold_key='val', root=os.getenv(
        "graph_files")+"/val"+"/data/")

    params = {
        "batch_size": 16,
        'shuffle': True
    }

    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    model = GATModel(dataset=test_set)  # For tensor dimensions

    model.eval()
    model.load_state_dict(torch.load(
        "GAT/weights/activation/model730.pth"))

    # Get the Predictions with Scores
    prec, symps, p = predict()
    print("Symptoms: ", symps)
    print("Confidence: ", p)
    print("Precision@k: ", prec)
