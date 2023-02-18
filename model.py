import code
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.utils as pyg_utils
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.norm import BatchNorm


# Taken and adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
class VariationalGCNEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels:int,
            use_batch_norm: bool,
            num_layers: int=2,
            name: str=None):
        super(VariationalGCNEncoder, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        modules = []
        
        for i in range(self.num_layers):
            if i == 0:
                modules.append(
                    (GCNConv(in_channels, hidden_channels, cached=False),
                            'x, edge_index -> h')
                )
            else:
                modules.append(
                    (GCNConv(hidden_channels, hidden_channels, cached=False),
                            'h, edge_index -> h')
                )
            if self.use_batch_norm:
                modules.append(BatchNorm(hidden_channels))
            modules.append(nn.ReLU(inplace=True))

        self.convs = Sequential('x, edge_index', modules)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=False)
        self.conv_logvar = GCNConv(hidden_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = self.convs(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class GraphModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('Graph representation learning using the VGAE framework.')

        self.vgae = VGAE(
            encoder=VariationalGCNEncoder(
                in_channels=args.num_features,
                hidden_channels=args.graph_latent_dim,
                out_channels=args.node_latent_dim,
                num_layers=args.num_layers,
                use_batch_norm=args.use_batch_norm,
            ),
            decoder=InnerProductDecoder()

        )

    def forward(self, x, edge_index):
        z = self.vgae.encode(x, edge_index)
        adj_pred = self.vgae.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, edge_index):
        num_nodes = x.size(0)
        z = self.vgae.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.vgae.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        edge_index_tmp, _ = pyg_utils.remove_self_loops(edge_index)
        edge_index_tmp, _ = pyg_utils.add_self_loops(edge_index_tmp)

        neg_edge_index = pyg_utils.negative_sampling(edge_index_tmp, z.size(0),
                                                     pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.vgae.decoder(z, neg_edge_index, sigmoid=True)
                              + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.vgae.kl_loss()

        return pos_loss + neg_loss + kl_loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index,
                    test_neg_edge_index):
        with torch.no_grad():
            z = self.vgae.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = \
            self.vgae.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score

    def get_mu(self, x, edge_index):
        mu, logvar = self.vgae.encoder.forward(x, edge_index)
        return mu
