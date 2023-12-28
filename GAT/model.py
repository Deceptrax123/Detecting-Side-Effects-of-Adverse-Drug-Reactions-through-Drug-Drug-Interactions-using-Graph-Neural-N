import torch
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_max_pool, GraphNorm
from torch.nn import Module, BatchNorm1d, Linear, LeakyReLU, Dropout1d, ReLU
from torch.nn.functional import sigmoid


class GATModel(Module):
    def __init__(self, dataset):
        super(GATModel, self).__init__()

        # Reactant 1
        self.x_attention1 = GATv2Conv(in_channels=dataset[0].x_s.size(1), out_channels=128,
                                      edge_dim=9, heads=8, dropout=0.4)

        self.gn1 = GraphNorm(in_channels=128*8)

        self.x_attention2 = GATv2Conv(
            in_channels=128*8, out_channels=256, edge_dim=9, dropout=0.4, heads=4)

        self.gn2 = GraphNorm(in_channels=256*4)

        self.x_attention3 = GATv2Conv(
            in_channels=256*4, out_channels=512, edge_dim=9, dropout=0.4)
        self.gn3 = GraphNorm(in_channels=512)

        # Reactant 2
        self.y_attention1 = GATv2Conv(in_channels=dataset[0].x_t.size(1), out_channels=128,
                                      edge_dim=9, heads=8, dropout=0.4)

        self.gn4 = GraphNorm(in_channels=128*8)
        self.lr4 = ReLU()
        self.y_attention2 = GATv2Conv(in_channels=128*8, out_channels=256,
                                      edge_dim=9, dropout=0.4, heads=4)
        self.gn5 = GraphNorm(in_channels=256*4)

        self.y_attention3 = GATv2Conv(
            in_channels=256*4, out_channels=512, edge_dim=9, dropout=0.4)
        self.gn6 = GraphNorm(512)

        # Linear Layers
        self.linear1 = Linear(512, out_features=1024)
        self.dp1 = Dropout1d(0.5)
        self.bn1 = BatchNorm1d(num_features=1024)

        self.linear2 = Linear(in_features=1024, out_features=1317)

    def forward(self, x, xs_batch, xt_batch):
        # graph1
        x1 = self.x_attention1(x.x_s, x.edge_index_s,
                               edge_attr=x.edge_attr_s).relu()
        x1 = self.gn1(x1)
        x2 = self.x_attention2(
            x1, x.edge_index_s, edge_attr=x.edge_attr_s).relu()
        x2 = self.gn2(x2)
        x3 = self.x_attention3(
            x2, x.edge_index_s, edge_attr=x.edge_attr_s).relu()
        x3 = self.gn3(x3)
        xs = global_mean_pool(x3, batch=xs_batch)

        # graph2
        x4 = self.y_attention1(x.x_t, x.edge_index_t,
                               edge_attr=x.edge_attr_t).relu()
        x4 = self.gn4(x4)
        x5 = self.y_attention2(
            x4, x.edge_index_t, edge_attr=x.edge_attr_t).relu()
        x5 = self.gn5(x5)
        x6 = self.y_attention3(
            x5, x.edge_index_t, edge_attr=x.edge_attr_t).relu()
        x6 = self.gn6(x6)
        xt = global_mean_pool(x6, batch=xt_batch)

        # Aggregates
        x = torch.add(xs, xt)

        # Classifier
        x = self.linear1(xs)
        x = self.bn1(x)
        x = self.lr7(x)
        x = self.linear2(x)

        return x, sigmoid(x)
