import torch
from torch_geometric.nn import GATConv, Linear, global_mean_pool
from torch.nn import Module, Sigmoid


class GATModel(Module):
    def __init__(self, dataset):
        super(GATModel, self).__init__()

        # Reactant 1
        self.x_attention1 = GATConv(in_channels=dataset[0].x_s.size(1), out_channels=dataset[0].x_s.size(1)//2,
                                    edge_dim=dataset[0].edge_index_s.size(1))

        self.x_attention2 = GATConv(in_channels=dataset[0].x_s.size(
            1)//2, out_channels=dataset[0].x_s.size(1)//4, edge_dim=dataset[0].edge_index_s.size(1))

        # Reactant 2
        self.y_attention1 = GATConv(in_channels=dataset[0].x_t.size(1), out_channels=dataset[0].x_t.size(1)//2,
                                    edge_dim=dataset[0].edge_index_t.size(1))
        self.y_attention2 = GATConv(in_channels=dataset[0].x_t.size(1)//2, out_channels=dataset[0].x_t.size(1)//4,
                                    edge_dim=dataset[0].edge_index_t.size(1))

        # Linear Layers
        self.linear = Linear(in_channels=dataset[0].x_t.size(
            1)//4, out_channels=dataset[0].y.size(0))  # classifier

        # Classsifier
        self.classifier = Sigmoid()

    def forward(self, x, xs_batch, xt_batch):
        # graph1
        x1 = self.x_attention1(x.x_s, x.edge_index_s)
        x2 = self.x_attention2(x1, x.edge_index_s)
        xs = global_mean_pool(x2, batch=xs_batch)

        # graph2
        x3 = self.y_attention1(x.x_t, x.edge_index_t)
        x4 = self.y_attention2(x3, x.edge_index_t)
        xt = global_mean_pool(x4, batch=xt_batch)

        # Aggregate
        x = torch.add(xs, xt)

        # Classifier
        out = self.linear(x)
        classifier = self.classifier(out)

        return classifier
