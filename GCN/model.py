import torch
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_max_pool, GraphNorm, GCNConv
from torch.nn import Module, BatchNorm1d, Linear, ReLU
from torch.nn.functional import sigmoid
class GCNModel(Module):
    def __init__(self, dataset):
        super(GCNModel, self).__init__()

        # Reactant 1
        self.x_layer1 = GCNConv(in_channels=dataset[0].x_s.size(1), out_channels=dataset[0].x_s.size(1)//2, dropout=0.4)
        
        self.relu1=ReLU()  

        self.gn1 = GraphNorm(in_channels=(dataset[0].x_s.size(1)//2))

        self.x_layer2 = GCNConv(in_channels=(dataset[0].x_s.size(1)//2), out_channels=dataset[0].x_s.size(1)//4, dropout=0.4)

        self.relu2=ReLU()

        self.gn2 = GraphNorm(in_channels=dataset[0].x_s.size(1)//4)

        self.x_layer3 = GCNConv(in_channels=(
            dataset[0].x_s.size(1)//4), out_channels=dataset[0].x_s.size(1)//8, dropout=0.4)

        self.relu3= ReLU()

        self.gn3 = GraphNorm(in_channels=dataset[0].x_t.size(1)//8)

        # Reactant 2
        self.y_layer1 = GCNConv(in_channels=dataset[0].x_t.size(1), out_channels=dataset[0].x_t.size(1)//2, dropout=0.4)

        self.relu4=ReLU()

        self.gn4 = GraphNorm(in_channels=(dataset[0].x_t.size(1)//2))

        self.y_layer2 = GCNConv(in_channels=(dataset[0].x_t.size(1)//2), out_channels=dataset[0].x_t.size(1)//4, dropout=0.4)

        self.relu5=ReLU()

        self.gn5 = GraphNorm(in_channels=dataset[0].x_s.size(1)//4)

        self.y_layer3 = GCNConv(in_channels=(
            dataset[0].x_t.size(1)//4), out_channels=dataset[0].x_t.size(1)//8, dropout=0.4)

        self.relu6=ReLU()

        self.gn6 = GraphNorm(in_channels=dataset[0].x_t.size(1)//8)

        # Linear Layers
        self.linear1 = Linear(in_features=(dataset[0].x_t.size(
            1)//8), out_features=329)
        self.bn1 = BatchNorm1d(num_features=329)

        self.linear2 = Linear(in_features=329, out_features=658)
        self.bn2 = BatchNorm1d(num_features=658)

        self.linear3 = Linear(in_features=658, out_features=1317)

    def forward(self, x, xs_batch, xt_batch):
        # graph1
        x1 = self.x_layer1(x.x_s, x.edge_index_s)
        x1 = self.relu1(x1)
        x1 = self.gn1(x1)
        x2 = self.x_layer2(x1, x.edge_index_s)
        x2 = self.relu2(x2)
        x2 = self.gn2(x2)
        x3 = self.x_layer3(x2, x.edge_index_s)
        x3 = self.relu3(x3)
        x3 = self.gn3(x3)
        xs = global_mean_pool(x3, batch=xs_batch)

        # graph2
        x4 = self.y_layer1(x.x_t, x.edge_index_t)
        x4 = self.relu4(x4)
        x4 = self.gn4(x4)
        x5 = self.y_layer2(x4, x.edge_index_t)
        x5 =self.relu5(x5)
        x5 = self.gn5(x5)
        x6 = self.y_layer3(x5, x.edge_index_t)
        x6 = self.gn6(x6)
        xt = global_mean_pool(x6, batch=xt_batch)

        # # Aggregate
        x = torch.add(xs, xt)

        # Classifier
        x = self.linear1(xs)
        x = self.bn1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.linear3(x)

        return x, sigmoid(x)
