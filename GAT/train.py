from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from Dataset.Molecule_dataset import MolecularGraphDataset
import torch
from model import GATModel
import torch_geometric.profile
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
from dotenv import load_dotenv
import os
import gc
import wandb


def train_epoch():
    epoch_loss = 0

    for step, graphs in enumerate(train_loader):

        predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)
        loss = loss_function(predictions, graphs.y)

        # Train the modle
        model.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        del graphs
        del predictions

    return epoch_loss/train_steps


def test_epoch():
    epoch_loss = 0

    for step, graphs in enumerate(test_loader):
        predictions = model(graphs, graphs.x_s_batch, graphs.x_t_batch)
        loss = loss_function(predictions, graphs.y)

        epoch_loss += loss.item()

        del graphs
        del predictions

    return epoch_loss/test_steps


def training_loop():
    for epoch in range(NUM_EPOCHS):

        model.train(True)
        train_loss = train_epoch()

        model.eval()

        with torch.no_grad():
            test_loss = test_epoch()

            print("Epoch {epoch}".format(epoch=epoch+1))
            print("Train Loss: {loss}".format(loss=train_loss))
            print("Test Loss: {loss}".format(loss=test_loss))

            wandb.log({
                "Training Loss": train_loss,
                "Testing Loss": test_loss
            })


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

    test_set = MolecularGraphDataset(test_fold, root=os.getenv(
        "graph_files")+"/fold3"+"/data/", start=15000)
    train_set = ConcatDataset([train_set1, train_set2])

    train_loader = DataLoader(train_set, **params, follow_batch=['x_s', 'x_t'])
    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    # The dataset is only for feature shape reference, no
    model = GATModel(dataset=train_set)
    # actual dataset is passed.

    NUM_EPOCHS = 10
    LR = 0.001
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)

    train_steps = (len(train_set)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test_set)+params['batch_size']-1)//params['batch_size']

    loss_function = nn.BCEWithLogitsLoss()

    training_loop()
