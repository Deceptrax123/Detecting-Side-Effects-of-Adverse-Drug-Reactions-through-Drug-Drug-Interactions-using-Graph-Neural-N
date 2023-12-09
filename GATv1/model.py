import torch
from torch_geometric.nn import GATConv, Linear, global_mean_pool
from torch.nn import Module, Sigmoid, ReLU

# No edge features were used.


class GATModel(Module):
    def __init__(self, dataset):
        super(GATModel, self).__init__()

        # Reactant 1
        self.x_attention1 = GATConv(in_channels=dataset[0].x_s.size(1), out_channels=dataset[0].x_s.size(1)//2,
                                    edge_dim=dataset[0].edge_index_s.size(1))

        self.relu1 = ReLU()

        self.x_attention2 = GATConv(in_channels=dataset[0].x_s.size(
            1)//2, out_channels=dataset[0].x_s.size(1)//4, edge_dim=dataset[0].edge_index_s.size(1))

        self.relu2 = ReLU()

        # Reactant 2
        self.y_attention1 = GATConv(in_channels=dataset[0].x_t.size(1), out_channels=dataset[0].x_t.size(1)//2,
                                    edge_dim=dataset[0].edge_index_t.size(1))

        self.relu3 = ReLU()
        self.y_attention2 = GATConv(in_channels=dataset[0].x_t.size(1)//2, out_channels=dataset[0].x_t.size(1)//4,
                                    edge_dim=dataset[0].edge_index_t.size(1))

        self.relu4 = ReLU()

        # Linear Layers
        self.linear = Linear(in_channels=dataset[0].x_t.size(
            1)//4, out_channels=dataset[0].y.size(0))  # classifier

        # Classsifier
        self.classifier = Sigmoid()

    def forward(self, x, xs_batch, xt_batch):
        # graph1
        x1 = self.x_attention1(x.x_s, x.edge_index_s)
        x1 = self.relu1(x1)
        x2 = self.x_attention2(x1, x.edge_index_s)
        x2 = self.relu2(x2)
        xs = global_mean_pool(x2, batch=xs_batch)

        # graph2
        x3 = self.y_attention1(x.x_t, x.edge_index_t)
        x3 = self.relu3(x3)
        x4 = self.y_attention2(x3, x.edge_index_t)
        x4 = self.relu4(x4)
        xt = global_mean_pool(x4, batch=xt_batch)

        # Aggregate
        x = torch.add(xs, xt)

        # Classifier
        out = self.linear(x)
        classifier = self.classifier(out)

        return classifier
