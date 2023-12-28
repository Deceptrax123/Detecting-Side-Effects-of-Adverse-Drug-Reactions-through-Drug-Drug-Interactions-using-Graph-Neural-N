import torch
from Dataset.Molecule_dataset import MolecularGraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Dataset.test_molecule_dataset import TestMolecularGraphDataset
from Metrics.metrics import classification_metrics, topk_precision
import torch
from encoder import DrugEncoder
from model import SSLModel
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
    f1s = list()
    over_precisions = list()
    for step, graphs in enumerate(test_loader):
        logits, predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        # precision, topk_labels, score = topk_precision(
        # predictions, graphs.y.int(), k=1)

        # top_symptoms = label_map_target(topk_labels)
        _, f, p = classification_metrics(predictions, graphs.y)

        # precisions.append(precision)
        # labels.append(top_symptoms)
        # scores.append(score)
        f1s.append(f)
        over_precisions.append(p)

    return sum(f1s)/len(f1s), sum(over_precisions)/len(over_precisions)


if __name__ == '__main__':

    load_dotenv(".env")
    test_set = TestMolecularGraphDataset(fold_key='val', root=os.getenv(
        "graph_files")+"/val"+"/data/")

    params = {
        "batch_size": 16,
        'shuffle': True
    }

    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    # Get Models
    r1_enc = DrugEncoder(in_features=test_set[0].x_s.size(1))
    r2_enc = DrugEncoder(in_features=test_set[0].x_t.size(1))

    model = SSLModel(r1_enc=r1_enc, r2_enc=r2_enc)  # For tensor dimensions

    model.eval()
    model.load_state_dict(torch.load(
        "MLP-SSL/weights/model400.pth"))

    # Get the Predictions with Scores
    f1, cp = predict()

    print("Overall Precision: ", cp)
    print("Overall F1: ", f1)
