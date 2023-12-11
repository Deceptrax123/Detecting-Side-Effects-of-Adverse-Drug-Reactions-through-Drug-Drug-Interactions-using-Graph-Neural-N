from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Dataset.Molecule_dataset import MolecularGraphDataset
from Metrics.metrics import classification_metrics, topk_precision
import torch
from model import GCNModel
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
from dotenv import load_dotenv
import os
import numpy as np
import gc
import wandb


def compute_weights(y_sample):
    # get counts of 1
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

    inverse = 1/weights
    inverse = inverse.astype(np.float32)

    return inverse


def train_epoch():
    epoch_loss = 0

    accs = list()

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
        weighted_accuracy = classification_metrics(
            preds_threshold.int(), graphs.y.int())

        accs.append(weighted_accuracy)

        del graphs
        del predictions

    return epoch_loss/train_steps, sum(accs)/len(accs)


def test_epoch():
    epoch_loss = 0
    accs = list()

    for step, graphs in enumerate(test_loader):
        # weights = torch.from_numpy(compute_weights(graphs.y))
        logits, predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)

        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(logits, graphs.y)

        epoch_loss += loss.item()

        preds_threshold = torch.where(predictions > 0.5, 1,
                                      0)

        # Compute Test Metrics
        accuracy = classification_metrics(
            preds_threshold.int(), graphs.y.int())

        accs.append(accuracy)

        del graphs
        del predictions

    return epoch_loss/test_steps, sum(accs)/len(accs)


def training_loop():
    for epoch in range(NUM_EPOCHS):

        model.train(True)
        train_loss, train_acc = train_epoch()

        model.eval()

        with torch.no_grad():
            test_loss, test_acc = test_epoch()

            print("Epoch {epoch}".format(epoch=epoch+1))
            print("Train Loss: {loss}".format(loss=train_loss))
            print("Test Loss: {loss}".format(loss=test_loss))

            print("Train Metrics")
            print("Train Accuracy:{acc}".format(acc=train_acc))

            print("Test Metrics")
            print("Test Accuracy:{acc}".format(acc=test_acc))

            wandb.log({
                "Training Loss": train_loss,
                "Testing Loss": test_loss,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,

            })

            #if (epoch+1) % 10 == 0:
               # weights_path = "GCN/weights/activation/model{epoch}.pth".format(
                  #  epoch=epoch+1)

                #torch.save(model.state_dict(), weights_path)

            if (epoch + 1) % 10 == 0:
                weights_dir = "GCN/weights/train_fold_12/head_1/"
                os.makedirs(weights_dir, exist_ok=True)  # Create directory if it doesn't exist
                weights_path = os.path.join(weights_dir, f"model{epoch + 1}.pth")
                torch.save(model.state_dict(), weights_path)



if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    load_dotenv(".env")

    # Set the training and testing folds
    train_folds = ['fold1', 'fold2', 'fold3',
                   'fold4', 'fold5', 'fold6', 'fold7']
    test_folds = ['fold8']

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
    train_set7 = MolecularGraphDataset(fold_key=train_folds[6], root=os.getenv("graph_files")+"/fold7/"
                                       + "/data/", start=45000)

    test_set = MolecularGraphDataset(fold_key=test_folds[0], root=os.getenv(
        "graph_files")+"/fold8"+"/data/", start=52500)

    train_set = ConcatDataset(
        [train_set1, train_set2, train_set3, train_set4, train_set5, train_set6, train_set7])

    train_loader = DataLoader(train_set, **params, follow_batch=['x_s', 'x_t'])
    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    # The dataset is only for feature shape reference, no
    model = GCNModel(dataset=train_set)
    for m in model.modules():
        init_weights(m)

    # actual dataset is passed.

    NUM_EPOCHS = 10000
    LR = 0.001
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)

    train_steps = (len(train_set)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test_set)+params['batch_size']-1)//params['batch_size']

    # Metrics
    training_loop()
