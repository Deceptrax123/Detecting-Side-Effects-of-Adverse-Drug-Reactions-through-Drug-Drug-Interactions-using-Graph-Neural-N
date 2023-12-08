import torch
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_max_pool, GraphNorm
from torch.nn import Module, BatchNorm1d, Linear
from torch.nn.functional import sigmoid


class GATModel(Module):
    def __init__(self, dataset):
        super(GATModel, self).__init__()

        # Reactant 1
        self.x_attention1 = GATv2Conv(in_channels=dataset[0].x_s.size(1), out_channels=dataset[0].x_s.size(1)//2,
                                      edge_dim=9, heads=8, dropout=0.4)

        self.gn1 = GraphNorm(in_channels=(dataset[0].x_s.size(
            1)//2)*8)

        self.x_attention2 = GATv2Conv(in_channels=(dataset[0].x_s.size(
            1)//2)*8, out_channels=dataset[0].x_s.size(1)//4, edge_dim=9, dropout=0.4)

        self.gn2 = GraphNorm(in_channels=dataset[0].x_s.size(1)//4)

        # Reactant 2
        self.y_attention1 = GATv2Conv(in_channels=dataset[0].x_t.size(1), out_channels=dataset[0].x_t.size(1)//2,
                                      edge_dim=9, heads=8, dropout=0.4)

        self.gn3 = GraphNorm(in_channels=(dataset[0].x_s.size(
            1)//2)*8)
        self.y_attention2 = GATv2Conv(in_channels=(dataset[0].x_t.size(1)//2)*8, out_channels=dataset[0].x_t.size(1)//4,
                                      edge_dim=9, dropout=0.4)
        self.gn4 = GraphNorm(in_channels=dataset[0].x_s.size(1)//4)

        # Linear Layers
        self.linear1 = Linear(in_features=(dataset[0].x_t.size(
            1)//4), out_features=658)
        self.bn1 = BatchNorm1d(num_features=658)
        self.linear2 = Linear(in_features=658, out_features=1317)

    def forward(self, x, xs_batch, xt_batch):
        # graph1
        x1 = self.x_attention1(x.x_s, x.edge_index_s, edge_attr=x.edge_attr_s)
        x1 = self.gn1(x1)
        x2 = self.x_attention2(x1, x.edge_index_s, edge_attr=x.edge_attr_s)
        x2 = self.gn2(x2)
        xs = global_mean_pool(x2, batch=xs_batch)

        # graph2
        x3 = self.y_attention1(x.x_t, x.edge_index_t, edge_attr=x.edge_attr_t)
        x3 = self.gn3(x3)
        x4 = self.y_attention2(x3, x.edge_index_t, edge_attr=x.edge_attr_t)
        x4 = self.gn4(x4)
        xt = global_mean_pool(x4, batch=xt_batch)

        # # Aggregate
        x = torch.add(xs, xt)

        # Classifier
        x = self.linear1(xs)
        x = self.bn1(x)
        x = self.linear2(x)

        return x, sigmoid(x)
