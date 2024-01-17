import torch
from torch_geometric.nn import Linear, global_max_pool, GCNConv
from torch.nn import Dropout1d
import torch.nn.functional as F


class SSLModel(torch.nn.Module):
    def __init__(self, r1_enc, r2_enc):
        super(SSLModel, self).__init__()

        self.enc1 = r1_enc
        self.enc2 = r2_enc

        # Linear Layers
        self.l1 = Linear(in_channels=512, out_channels=1024)
        self.l2 = Linear(in_channels=1024, out_channels=1317)

        # Dropout
        self.dp = Dropout1d()

    def forward(self, x, xs_batch, xt_batch):

        enc_r1 = self.enc1(x.x_s, x.edge_index_s)
        enc_r2 = self.enc2(x.x_t, x.edge_index_t)

        r1_pooled = global_max_pool(enc_r1, batch=xs_batch)
        r2_pooled = global_max_pool(enc_r2, batch=xt_batch)

        z = torch.add(r1_pooled, r2_pooled)

        # Classifier
        z = self.l1(z).relu()
        z = self.dp(z)
        z = self.l2(z)

        return z, F.sigmoid(z)
