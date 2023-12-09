import torch
from torch_geometric.nn import global_mean_pool
from torch.nn import ReLU, Linear, Dropout1d, BatchNorm1d


class MLPModel(torch.nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()

        self.linear1 = Linear(in_features=86, out_features=43)
        self.dp1 = Dropout1d(p=0.5)
        self.bn1 = BatchNorm1d(43)
        self.relu1 = ReLU()

        self.linear2 = Linear(in_features=43, out_features=21)
        self.dp2 = Dropout1d()
        self.bn2 = BatchNorm1d(21)
        self.relu2 = ReLU()

        self.linear3 = Linear(in_features=21, out_features=10)
        self.dp3 = Dropout1d()
        self.bn3 = BatchNorm1d(10)
        self.relu3 = ReLU()

        self.linear4 = Linear(in_features=10, out_features=20)
        self.dp4 = Dropout1d()
        self.bn4 = BatchNorm1d(20)
        self.relu4 = ReLU()

        self.linear5 = Linear(in_features=20, out_features=40)
        self.dp5 = Dropout1d()
        self.bn5 = BatchNorm1d(40)
        self.relu5 = ReLU()

        self.linear6 = Linear(in_features=40, out_features=80)
        self.dp6 = Dropout1d()
        self.bn6 = BatchNorm1d(80)
        self.relu6 = ReLU()

        self.linear7 = Linear(in_features=80, out_features=1317)

    def forward(x, xs_batch, xt_batch):
        pass
