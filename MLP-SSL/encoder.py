from torch.nn import Module
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class DrugEncoder(Module):
    def __init__(self, in_features):
        super(DrugEncoder, self).__init__()

        self.gcn1 = GCNConv(
            in_channels=in_features, out_channels=128, normalize=True)
        self.gcn2 = GCNConv(
            in_channels=128, out_channels=256, normalize=True)
        self.gcn3 = GCNConv(
            in_channels=256, out_channels=512, normalize=True)

        self.conv_mu = GCNConv(in_channels=512, out_channels=256)
        self.conv_std = GCNConv(in_channels=512, out_channels=256)

    def forward(self, v, edge_index):
        x = self.gcn1(v, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = self.gcn3(x, edge_index).relu()

        return x
