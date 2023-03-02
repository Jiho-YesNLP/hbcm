import code
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.utils as pyg_utils
from torch_geometric.nn import Sequential, GCNConv, Linear
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.norm import BatchNorm


# Taken and adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 use_batch_norm, edge_index, num_layers=2, name=None):
        super(VariationalGCNEncoder, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers
        self.edge_index = edge_index

        modules = []
        
        for i in range(self.num_layers):
            if i == 0:
                modules.append(
                    (GCNConv(in_channels, hidden_channels),
                            'x, edge_index -> h')
                )
            else:
                modules.append(
                    (GCNConv(hidden_channels, hidden_channels),
                            'h, edge_index -> h')
                )
            if self.use_batch_norm:
                modules.append(BatchNorm(hidden_channels))
            modules.append(nn.ReLU(inplace=True))

        self.convs = Sequential('x, edge_index', modules)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x):
        x = self.convs(x, self.edge_index)
        return self.conv_mu(x, self.edge_index), \
            self.conv_logstd(x, self.edge_index)


class GraphModel(nn.Module):
    def __init__(self, args, edge_index, n):
        super().__init__()
        print('Graph representation learning using the VGAE framework.')

        self.edge_index = edge_index
        self.dadj = pyg_utils.to_dense_adj(self.edge_index, max_num_nodes=n)
        self.adj_norm = n * n / float((n * n - self.dadj.sum()) * 2)
        self.vgae = VGAE(
            encoder=VariationalGCNEncoder(
                # in_channels=args.num_features,
                in_channels=args.num_features,
                hidden_channels=args.graph_latent_dim,
                out_channels=args.node_latent_dim,
                num_layers=args.num_layers,
                use_batch_norm=args.use_batch_norm,
                edge_index=self.edge_index
            ),
            decoder=InnerProductDecoder()
        )

    def forward(self, x):
        z = self.vgae.encode(x)
        adj_pred = self.vgae.decoder.forward_all(z)
        return adj_pred

    def loss(self, a_pred):
        mu = self.vgae.__mu__
        logstd = self.vgae.__logstd__
        recon = F.binary_cross_entropy(a_pred.view(-1), self.dadj.view(-1))
        kl = -0.5 / a_pred.shape[0] * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)
        )
        return recon + kl

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index,
                    test_neg_edge_index):
        with torch.no_grad():
            z = self.vgae.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = \
            self.vgae.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score

    def get_mu(self, x):
        mu, logstd = self.vgae.encoder.forward(x)
        return mu

