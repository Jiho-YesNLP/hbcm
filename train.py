import code
import pickle
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from tensorboardX import SummaryWriter  # Tnesorflow monitoring tools

from HBCM.data import PubmedDataset

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = pyg_nn.GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
        
def train(epoch):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    
    writer.add_scalar("loss", loss.item(), epoch)
    
def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
    
    
if __name__ == '__main__':
    
    ds = PubmedDataset('data/')
    channels = 16
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', torch.cuda.is_available())
    
    # encoder: written by us; decoder: default (inner product)
    model = pyg_nn.GAE(Encoder(ds.num_features, channels)).to(dev)
    data = ds[0]
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    
    data = pyg_utils.train_test_split_edges(data)
    transform = RandomLinkSplit(is_undirected=True)
    transform(data)
    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # train
    for epoch in range(1, 201):
        train(epoch)
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        writer.add_scalar("AUC", auc, epoch)
        writer.add_scalar("AP", ap, epoch)
        if epoch % 10 == 0:
            print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
    code.interact(local=dict(locals(), **globals()))
        
    