import torch
from Dataset.Molecule_dataset import MolecularGraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Dataset.test_molecule_dataset import TestMolecularGraphDataset
from Dataset.Molecule_dataset import MolecularGraphDataset
from Metrics.metrics_test import classification_metrics
import torch
from model import SAGEConvModel
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

    f1s = list()
    over_precisions = list()
    aurocs = list()
    accs = list()
    recs=list()
    for step, graphs in enumerate(test_loader):
        logits, predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        # precision, topk_labels, score = topk_precision(
        # predictions, graphs.y.int(), k=1)

        # top_symptoms = label_map_target(topk_labels)

        acc, f, p, auroc,rec = classification_metrics(predictions, graphs.y.int())

        # precisions.append(precision)
        # labels.append(top_symptoms)
        # scores.append(score)
        accs.append(acc)
        f1s.append(f)
        over_precisions.append(p)
        aurocs.append(auroc)
        recs.append(rec)

    return sum(accs)/len(accs), sum(f1s)/len(f1s), sum(over_precisions)/len(over_precisions), sum(aurocs)/len(aurocs),sum(recs)/len(recs)


if __name__ == '__main__':

    load_dotenv(".env")
    test_set=MolecularGraphDataset(fold_key='fold7', root=os.getenv(
        "graph_files")+"/fold7"+"/data/",start=45000)

    params = {
        "batch_size": 16,
        'shuffle': True
    }

    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    model = SAGEConvModel(dataset=test_set)

    model.eval()
    model.load_state_dict(torch.load(
        "SAGE/weights/model380.pth"))

    # Get the Predictions with Scores
    acc, _, cp, auroc,recall = predict()

    print("Overall Precision: ", cp)
    print("Area under ROC: ", auroc)
    print("Accuracy: ", acc)
    print("recall: ",recall)
