import torch
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_max_pool, GraphNorm
from torch.nn import Module, BatchNorm1d, Linear, LeakyReLU, Dropout1d, ELU
from torch.nn.functional import sigmoid


class GATModel(Module):
    def __init__(self, dataset):
        super(GATModel, self).__init__()

        # Reactant 1
        self.x_attention1 = GATv2Conv(in_channels=dataset[0].x_s.size(1), out_channels=256,
                                      edge_dim=9, heads=4)
        self.lr1 = ELU()
        self.gn1 = GraphNorm(256*4)

        self.x_attention2 = GATv2Conv(
            in_channels=256*4, out_channels=256, edge_dim=9, heads=4)
        self.lr2 = ELU()
        self.gn2 = GraphNorm(256*4)

        self.x_attention3 = GATv2Conv(
            in_channels=256*4, out_channels=1317, edge_dim=9, heads=1)
        self.lr3 = ELU()

        # Reactant 2
        self.y_attention1 = GATv2Conv(in_channels=dataset[0].x_t.size(1), out_channels=256,
                                      edge_dim=9, heads=4)
        self.gn3 = GraphNorm(in_channels=(dataset[0].x_t.size(
            1)//2)*8)
        self.lr4 = ELU()

        self.y_attention2 = GATv2Conv(
            in_channels=256*4, out_channels=256, edge_dim=9, heads=4)
        self.gn4 = GraphNorm(in_channels=256*4)
        self.lr5 = ELU()

        self.y_attention3 = GATv2Conv(
            in_channels=256*4, out_channels=1317, edge_dim=9, heads=1)
        self.lr6 = ELU()

    def forward(self, x, xs_batch, xt_batch):
        # graph1
        x1 = self.x_attention1(x.x_s, x.edge_index_s, edge_attr=x.edge_attr_s)
        # x1 = self.gn1(x1)
        x1 = self.lr1(x1)
        x2 = self.x_attention2(x1, x.edge_index_s, edge_attr=x.edge_attr_s)
        # x2 = self.gn2(x2)
        x2 = self.lr2(x2)
        x3 = self.x_attention3(x2, x.edge_index_s, edge_attr=x.edge_attr_s)
        x3 = self.lr3(x3)
        xs = global_mean_pool(x3, batch=xs_batch)

        # graph2
        x4 = self.y_attention1(x.x_t, x.edge_index_t, edge_attr=x.edge_attr_t)
        # x4 = self.gn3(x4)
        x4 = self.lr4(x4)
        x5 = self.y_attention2(x4, x.edge_index_t, edge_attr=x.edge_attr_t)
        # x5 = self.gn4(x5)
        x5 = self.lr5(x5)
        x6 = self.y_attention3(x5, x.edge_index_t, edge_attr=x.edge_attr_t)
        x6 = self.lr6(x6)
        xt = global_mean_pool(x6, batch=xt_batch)

        # Aggregate
        x = torch.add(xs, xt)

        return x, sigmoid(x)
