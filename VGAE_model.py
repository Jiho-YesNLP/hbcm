import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
import os
from torch_geometric.utils import coalesce
from tensorboardX import SummaryWriter
import time
from datetime import datetime
import numpy as np
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from model import DeepVGAE
from config import parse_args



class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, args):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(enc_in_channels,
                                                          enc_hidden_channels,
                                                          enc_out_channels),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, edge_index):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        # edge_index_tmp, _ = remove_self_loops(edge_index)
        # edge_index_tmp, _ = add_self_loops(edge_index_tmp)

        neg_edge_index = negative_sampling(edge_index, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score

if __name__=='__main__':
    torch.manual_seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    db_file = 'C:/Users/14707/Documents/Conference_Paper/code/hbcm_main/data/edgelist.pkl'
    data = pickle.load(open(db_file, 'rb'))
    df = data['ent2idx']
    num_nodes = len(data['ent2idx'])
    x = torch.randn(len(data['ent2idx']), 64)

    # Convert the node degrees to a tensor and use it as the feature matrix x

    rows, cols = [], []
    for (u, v), nmi in data['edges'].items():
        if nmi>0.40:
            rows.append(u)
            cols.append(v)

    edge_index = torch.stack([torch.tensor(rows).to(torch.long),
                                    torch.tensor(cols).to(torch.long)], dim=0)
    edge_index = coalesce(edge_index, num_nodes=len(data['ent2idx']))
    data = Data(x=x, edge_index=edge_index)
    T_data = pyg_utils.train_test_split_edges(data)

    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    enc_in_channels = 64
    enc_hidden_channels =  32
    enc_out_channels = 16


    epoch =  100
    model = DeepVGAE(args)
    encoder_z, _ = model.encoder(x, edge_index)
    embeddings_z = encoder_z.detach().numpy()
# np.save('C:/Users/14707/Documents/Conference_Paper/code/hbcm_main/HBCM/embeddings/embeddings_z.npy', embeddings_z)

    # Extract the embeddings from mu
    model_1 = GCNEncoder(64,32,16)
    mu = model_1.gcn_mu(model_1.gcn_shared(x, edge_index), edge_index)
    embeddings_mu = mu.detach().numpy()
# np.save('C:/Users/14707/Documents/Conference_Paper/code/hbcm_main/HBCM/embeddings/embeddings_mu.npy', embeddings_mu)
# Extract the embeddings from logvar
    logvar = model_1.gcn_logvar(model_1.gcn_shared(x, edge_index), edge_index)
    embeddings_lv = logvar.detach().numpy()
# np.save('C:/Users/14707/Documents/Conference_Paper/code/hbcm_main/HBCM/embeddings/embeddings_lv.npy', embeddings_lv)
#finding tensor values for mesh and keyword
# tensor_values = []
# for key in df:
#     index = df[key]
#     tensor_value = embeddings[index]
#     tensor_values.append(tensor_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(T_data.x, T_data.train_pos_edge_index, edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            model.eval()
            roc_auc, ap = model.single_test(T_data.x,
                                            T_data.train_pos_edge_index,
                                            T_data.test_pos_edge_index,
                                            T_data.test_neg_edge_index)
            print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))

            
