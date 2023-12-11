import torch
from torch import nn
from torch_geometric.nn import GCNConv, GraphNorm


def initialize(model):
    for m in model.modules():
        if isinstance(m, (GCNConv, GraphNorm)):
            if m.bias.data is not None:
                m.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
