import torch
from torch_geometric.nn import Linear, global_max_pool, GCNConv
import torch.nn.functional as F


class SSLModel(torch.nn.Module):
    def __init__(self, r1_enc, r2_enc):
        super(SSLModel, self).__init__()

        self.enc1 = r1_enc
        self.enc2 = r2_enc

        # Reactant 1
        self.r1_gcn1 = GCNConv(
            in_channels=512, out_channels=1024, normalize=True)
        self.r1_gcn2 = GCNConv(
            in_channels=1024, out_channels=2048, normalize=True)

        # Reactant 2
        self.r2_gcn1 = GCNConv(
            in_channels=512, out_channels=1024, normalize=True)
        self.r2_gcn2 = GCNConv(
            in_channels=1024, out_channels=2048, normalize=True)

        # Linear Layers
        self.l1 = Linear(in_channels=2048, out_channels=1024)
        self.l2 = Linear(in_channels=1024, out_channels=1317)

    def forward(self, x, xs_batch, xt_batch):

        enc_r1 = self.enc1(x.x_s, x.edge_index_s)
        enc_r2 = self.enc2(x.x_t, x.edge_index_t)

        # Reactant 1
        z_r1 = self.r1_gcn1(enc_r1, x.edge_index_s).relu()
        z_r1 = self.r1_gcn2(z_r1, x.edge_index_s).relu()

        # Reactant 2
        z_r2 = self.r2_gcn1(enc_r2, x.edge_index_t).relu()
        z_r2 = self.r2_gcn2(z_r2, x.edge_index_t).relu()

        r1_pooled = global_max_pool(z_r1, batch=xs_batch)
        r2_pooled = global_max_pool(z_r2, batch=xt_batch)

        z = torch.add(r1_pooled, r2_pooled)

        # Classifier
        z = self.l1(z).relu()
        z = self.l2(z)

        return z, F.sigmoid(z)
