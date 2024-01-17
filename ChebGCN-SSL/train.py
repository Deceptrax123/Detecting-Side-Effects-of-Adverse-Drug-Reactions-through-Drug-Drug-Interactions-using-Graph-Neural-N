from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Dataset.Molecule_dataset import MolecularGraphDataset
from Metrics.metrics import classification_metrics, topk_precision
import torch
from model import SSLModel
from encoder import SpectralDrugEncoder
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
from dotenv import load_dotenv
import os
import numpy as np
import gc
import wandb


def train_epoch():
    epoch_loss = 0

    accs = list()
    f1_micro = list()
    precs = list()

    for step, graphs in enumerate(train_loader):

        # weights = torch.from_numpy(compute_weights(graphs.y))
        logits, predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        # Train Model
        model.zero_grad()
        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(logits, graphs.y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        preds_threshold = torch.where(predictions > 0.5, 1,
                                      0)

        # Weighted accuracy if 0.5 is given as Hard Threshold
        weighted_accuracy, f1, precision = classification_metrics(
            preds_threshold.int(), graphs.y.int())

        accs.append(weighted_accuracy)
        f1_micro.append(f1)
        precs.append(precision)

        del graphs
        del predictions

    return epoch_loss/train_steps, sum(accs)/len(accs), sum(f1_micro)/len(f1_micro), sum(precs)/len(precs)


def test_epoch():
    epoch_loss = 0
    accs = list()
    f1_micro = list()
    precs = list()

    for step, graphs in enumerate(test_loader):
        # weights = torch.from_numpy(compute_weights(graphs.y))
        logits, predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(logits, graphs.y)

        epoch_loss += loss.item()

        preds_threshold = torch.where(predictions > 0.5, 1,
                                      0)

        # Compute Test Metrics
        accuracy, f1, precision = classification_metrics(
            preds_threshold.int(), graphs.y.int())

        accs.append(accuracy)
        f1_micro.append(f1)
        precs.append(precision)

        del graphs
        del predictions

    return epoch_loss/test_steps, sum(accs)/len(accs), sum(f1_micro)/len(f1_micro), sum(precs)/len(precs)


def training_loop():
    for epoch in range(NUM_EPOCHS):

        model.train(True)
        train_loss, train_acc, train_f1, train_precision = train_epoch()

        model.eval()

        with torch.no_grad():
            test_loss, test_acc, test_f1, test_precision = test_epoch()

            print("Epoch {epoch}".format(epoch=epoch+1))
            print("Train Loss: {loss}".format(loss=train_loss))
            print("Test Loss: {loss}".format(loss=test_loss))

            print("Train Metrics")
            print("Train Accuracy:{acc}".format(acc=train_acc))
            print("Train F1: {acc}".format(acc=train_f1))
            print("Train Precision: {acc}".format(acc=train_precision))

            print("Test Metrics")
            print("Test Accuracy:{acc}".format(acc=test_acc))
            print("Test F1: {acc}".format(acc=test_f1))
            print("Test Precision: {acc}".format(acc=test_precision))

            wandb.log({
                "Training Loss": train_loss,
                "Testing Loss": test_loss,
                "Train Accuracy": train_acc,
                "Train Precision": train_precision,
                "Test Precision": test_precision,
                "Test Accuracy": test_acc,
                "Test F1 micro": test_f1,
                "Train F1": train_f1

            })

            if (epoch+1) % 10 == 0:
                weights_path = "ChebGCN-SSL/weights/model{epoch}.pth".format(
                    epoch=epoch+1)

                torch.save(model.state_dict(), weights_path)

        # Update learning rate
        scheduler.step(test_loss)


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    load_dotenv(".env")

    # Set the training and testing folds
    train_folds = ['fold1', 'fold2', 'fold3',
                   'fold4', 'fold5', 'fold6']
    test_folds = ['fold7', 'fold8']

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }

    wandb.init(
        project="Drug Interaction",
        config={
            "Architecture": "Graph Attention Network",
            "Dataset": "TDC Dataset TWOSIDES",
        }
    )

    train_set1 = MolecularGraphDataset(
        fold_key=train_folds[0], root=os.getenv("graph_files")+"/fold1"+"/data/", start=0)
    train_set2 = MolecularGraphDataset(fold_key=train_folds[1], root=os.getenv("graph_files")+"/fold2/"
                                       + "/data/", start=7500)
    train_set3 = MolecularGraphDataset(fold_key=train_folds[2], root=os.getenv("graph_files")+"/fold3/"
                                       + "/data/", start=15000)
    train_set4 = MolecularGraphDataset(fold_key=train_folds[3], root=os.getenv("graph_files")+"/fold4/"
                                       + "/data/", start=22500)
    train_set5 = MolecularGraphDataset(fold_key=train_folds[4], root=os.getenv("graph_files")+"/fold5/"
                                       + "/data/", start=30000)
    train_set6 = MolecularGraphDataset(fold_key=train_folds[5], root=os.getenv("graph_files")+"/fold6/"
                                       + "/data/", start=37500)

    test_set1 = MolecularGraphDataset(fold_key=test_folds[0], root=os.getenv("graph_files")+"/fold7/"
                                      + "/data/", start=45000)
    test_set2 = MolecularGraphDataset(fold_key=test_folds[1], root=os.getenv(
        "graph_files")+"/fold8"+"/data/", start=52500)

    train_set = ConcatDataset(
        [train_set1, train_set2, train_set3, train_set4, train_set5, train_set6])

    test_set = ConcatDataset([test_set1, test_set2])

    train_loader = DataLoader(train_set, **params, follow_batch=['x_s', 'x_t'])
    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    # Get Models
    r1_enc = SpectralDrugEncoder(in_features=train_set[0].x_s.size(1))
    r1_enc.load_state_dict(torch.load("ChebGCN-SSL/r1_encoder.pth"))

    r2_enc = SpectralDrugEncoder(in_features=train_set[0].x_t.size(1))
    r2_enc.load_state_dict(torch.load("ChebGCN-SSL/r2_encoder.pth"))

    model = SSLModel(r1_enc=r1_enc, r2_enc=r2_enc)

    NUM_EPOCHS = 10000
    LR = 0.005
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min')

    train_steps = (len(train_set)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test_set)+params['batch_size']-1)//params['batch_size']

    # Metrics
    training_loop()
