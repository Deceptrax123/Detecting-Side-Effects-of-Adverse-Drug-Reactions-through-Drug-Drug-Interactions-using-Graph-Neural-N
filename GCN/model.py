import torch
from torch_geometric.nn import global_mean_pool, global_max_pool, GraphNorm, GCNConv
from torch.nn import Module, BatchNorm1d, Linear, ReLU
from torch.nn.functional import sigmoid,relu


class GCNModel(Module):
    def __init__(self, dataset):
        super(GCNModel, self).__init__()

        # Reactant 1
        self.x_layer1 = GCNConv(in_channels=dataset[0].x_s.size(
            1), out_channels=128, dropout=0.4)

        self.gn1 = GraphNorm(in_channels=128)

        self.x_layer2 = GCNConv(in_channels=128, out_channels=256, dropout=0.4)

        self.gn2 = GraphNorm(in_channels=256)

        self.x_layer3 = GCNConv(in_channels=256, out_channels=512, dropout=0.4)

        self.gn3 = GraphNorm(in_channels=512)

        # Reactant 2
        self.y_layer1 = GCNConv(in_channels=dataset[0].x_s.size(
            1), out_channels=128, dropout=0.4)

        self.gn4 = GraphNorm(in_channels=128)

        self.y_layer2 = GCNConv(in_channels=128, out_channels=256, dropout=0.4)

        self.gn5 = GraphNorm(in_channels=256)

        self.y_layer3 = GCNConv(in_channels=256, out_channels=512, dropout=0.4)

        self.gn6 = GraphNorm(in_channels=512)

        # Linear Layers
        self.linear1 = Linear(in_features=512, out_features=1024)

        self.bn1 = BatchNorm1d(num_features=1024)

        self.linear2 = Linear(in_features=1024, out_features=1317)

    def forward(self, x, xs_batch, xt_batch):
        # graph1
        x1 = self.x_layer1(x.x_s, x.edge_index_s).relu()
        x1 = self.gn1(x1)
        x2 = self.x_layer2(x1, x.edge_index_s).relu()
        x2 = self.gn2(x2)
        x3 = self.x_layer3(x2, x.edge_index_s).relu()
        x3 = self.gn3(x3)
        xs = global_mean_pool(x3, batch=xs_batch)

        # graph2
        x4 = self.y_layer1(x.x_t, x.edge_index_t).relu()
        x4 = self.gn4(x4)
        x5 = self.y_layer2(x4, x.edge_index_t).relu()
        x5 = self.gn5(x5)
        x6 = self.y_layer3(x5, x.edge_index_t).relu()
        x6 = self.gn6(x6)
        xt = global_mean_pool(x6, batch=xt_batch)

        # # Aggregate
        x = torch.add(xs, xt)

        # Classifier
        x = self.linear1(xs).relu()
        x = self.bn1(x)
        x = self.linear2(x)

        return x, sigmoid(x)
