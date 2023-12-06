from torch_geometric.loader import DataLoader
from Dataset.Molecule_dataset import MolecularGraphDataset
from Metrics.metrics import classification_metrics
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


def compute_weights(y_sample):
    # get counts of 1s
    labels = y_sample.size(1)

    counts = list()
    for i in range(labels):
        region = y_sample[:, i]

        ones = (region == 1.).sum()

        if ones == 0:
            ones = np.inf
        counts.append(ones)

    total_features = y_sample.size(0)*y_sample.size(1)

    counts = np.array(counts)
    weights = counts/total_features

    inverse = (1/weights)
    inverse = inverse.astype(np.float32)
    return inverse


def train_epoch():
    epoch_loss = 0

    accs = list()
    precisions = list()
    f1s = list()
    recalls = list()

    for step, graphs in enumerate(train_loader):

        weights = torch.from_numpy(compute_weights(graphs.y))
        predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        loss_function = nn.BCELoss(weight=weights)
        loss = loss_function(predictions, graphs.y)

        # Train the model
        model.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        preds_threshold = torch.tensor(torch.where(predictions > 0.5, 1,
                                                   0), dtype=torch.int64)

        precision, f1, accuracy, rec = classification_metrics(
            preds_threshold, graphs.y)

        precisions.append(precision)
        f1s.append(f1)
        accs.append(accuracy)
        recalls.append(rec)

        del graphs
        del predictions

    return epoch_loss/train_steps, sum(precisions)/len(precisions), sum(f1s)/len(f1s), sum(accs)/len(accs), sum(recalls)/len(recalls)


def test_epoch():
    epoch_loss = 0
    precisions = list()
    f1s = list()
    accs = list()
    recalls = list()

    for step, graphs in enumerate(test_loader):
        weights = torch.from_numpy(compute_weights(graphs.y))
        predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        loss_function = nn.BCELoss(weight=weights)
        loss = loss_function(predictions, graphs.y)

        epoch_loss += loss.item()

        preds_threshold = torch.tensor(torch.where(predictions > 0.5, 1,
                                                   0), dtype=torch.int64)

        # Compute Test Metrics
        precision, f1, accuracy, rec = classification_metrics(
            preds_threshold, graphs.y)

        precisions.append(precision)
        f1s.append(f1)
        accs.append(accuracy)
        recalls.append(rec)

        del graphs
        del predictions

    return epoch_loss/test_steps, sum(precisions)/len(precisions), sum(f1s)/len(f1s), sum(accs)/len(accs), sum(recalls)/len(recalls)


def training_loop():
    for epoch in range(NUM_EPOCHS):

        model.train(True)
        train_loss, train_prec, train_f1, train_acc, train_rec = train_epoch()

        model.eval()

        with torch.no_grad():
            test_loss, test_prec, test_f1, test_acc, test_rec = test_epoch()

            print("Epoch {epoch}".format(epoch=epoch+1))
            print("Train Loss: {loss}".format(loss=train_loss))
            print("Test Loss: {loss}".format(loss=test_loss))

            print("Train Metrics")
            print("Train Accuracy:{acc}".format(acc=train_acc))
            print("Train F1:{f1}".format(f1=train_f1))
            print("Train Precision:{precision}".format(precision=train_prec))
            print("Train Recall:{rec}".format(rec=train_rec))

            print("Test Metrics")
            print("Test Accuracy:{acc}".format(acc=test_acc))
            print("Test F1: {f1}".format(f1=test_f1))
            print("Test Precision:{precision}".format(precision=test_prec))
            print("Test Recall:{rec}".format(rec=test_rec))

            wandb.log({
                "Training Loss": train_loss,
                "Testing Loss": test_loss,
                "Train Accuracy": train_acc,
                "Train F1": train_f1,
                "Train Recall": train_rec,
                "Train Precision": train_prec,
                "Test Accuracy": test_acc,
                "Test F1": test_f1,
                "Test Precision": test_prec,
                "Test Recall": test_rec
            })

            if (epoch+1) % 10 == 0:
                weights_path = "GAT/weights/train_fold_12/head_1/model{epoch}.pth".format(
                    epoch=epoch+1)

                torch.save(model.state_dict(), weights_path)


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    load_dotenv(".env")

    # Set the training and testing folds
    train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
    test_folds = ['fold5', 'fold6']

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }

    wandb.init(
        project="Drug Interaction",
        config={
            "Architecture": "Graph Attention Network",
            "Dataset": "TDC Dataset TWOSIDES"
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

    test_set1 = MolecularGraphDataset(fold_key=test_folds[0], root=os.getenv(
        "graph_files")+"/fold5"+"/data/", start=30000)
    test_set2 = MolecularGraphDataset(fold_key=test_folds[1], root=os.getenv(
        "graph_files")+"/fold6"+"/data/", start=37500)

    train_set = ConcatDataset([train_set1, train_set2, train_set3, train_set4])
    test_set = ConcatDataset([test_set1, test_set2])

    train_loader = DataLoader(train_set, **params, follow_batch=['x_s', 'x_t'])
    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    # The dataset is only for feature shape reference, no
    model = GATModel(dataset=train_set)
    # actual dataset is passed.

    NUM_EPOCHS = 1
    LR = 0.001
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)

    train_steps = (len(train_set)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test_set)+params['batch_size']-1)//params['batch_size']

    # Metrics
    training_loop()
