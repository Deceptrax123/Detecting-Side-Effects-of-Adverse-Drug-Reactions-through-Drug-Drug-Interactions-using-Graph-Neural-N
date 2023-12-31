import torch
from Dataset.Molecule_dataset import MolecularGraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Dataset.test_molecule_dataset import TestMolecularGraphDataset
from Metrics.metrics_test import classification_metrics
import torch
import torch.nn.functional as F
from Cheb.encoder import SpectralDrugEncoder
from Cheb.model import SSLModel as ChebModel
from GCN_SSL.encoder import DrugEncoder
from GCN_SSL.model import SSLModel as GCNModel
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
    recs = list()
    for step, graphs in enumerate(test_loader):
        logits_1, _ = model_cheb(
            graphs, graphs.x_s_batch, graphs.x_t_batch)
        logits_2, _ = model_gcn(
            graphs, graphs.x_s_batch, graphs.x_t_batch)

        logits = torch.add(0.80*logits_1, 0.20*logits_2)

        predictions = F.sigmoid(logits)

        # precision, topk_labels, score = topk_precision(
        # predictions, graphs.y.int(), k=1)

        # top_symptoms = label_map_target(topk_labels)

        acc, f, p, auroc, rec = classification_metrics(
            predictions, graphs.y.int())

        # precisions.append(precision)
        # labels.append(top_symptoms)
        # scores.append(score)
        accs.append(acc)
        f1s.append(f)
        over_precisions.append(p)
        aurocs.append(auroc)
        recs.append(rec)

    return sum(accs)/len(accs), sum(f1s)/len(f1s), sum(over_precisions)/len(over_precisions), sum(aurocs)/len(aurocs), sum(recs)/len(recs)


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
    r1_enc_cheb = SpectralDrugEncoder(in_features=test_set[0].x_s.size(1))
    r2_enc_cheb = SpectralDrugEncoder(in_features=test_set[0].x_t.size(1))

    model_cheb = ChebModel(r1_enc=r1_enc_cheb, r2_enc=r2_enc_cheb)

    model_cheb.eval()
    model_cheb.load_state_dict(torch.load(
        "cheb_ssl_unweighted.pth"))

    r1_enc_gcn = DrugEncoder(in_features=test_set[0].x_s.size(1))
    r2_enc_gcn = DrugEncoder(in_features=test_set[0].x_t.size(1))

    model_gcn = GCNModel(r1_enc=r1_enc_gcn, r2_enc=r2_enc_gcn)

    model_gcn.eval()
    model_gcn.load_state_dict(torch.load(
        "GCN-SSL/weights/model410.pth"))

    acc, _, cp, auroc, recall = predict()

    print("Overall Precision: ", cp)
    print("Area under ROC: ", auroc)
    print("Accuracy: ", acc)
    print("Recall: ", recall)
