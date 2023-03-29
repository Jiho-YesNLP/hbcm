import code
import pickle
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import random
from model_new import GraphModel
from data_load import DBLPDataset
#from data_loading import DBLPDataset
from torch_geometric.utils import subgraph
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

#def find_topk(name, mu, ent2idx, idx2ent, k=8):
    #if name not in ent2idx:
        #print('name not found in dictionary')
        #return None
    
    #nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(mu.detach().numpy())

    #query_idx = ent2idx[name]
    #query_vector = mu[query_idx].unsqueeze(0).detach().numpy()
    
    #_, indices = nbrs.kneighbors(query_vector)

    #for idx in indices[0][1:]:
        #print(idx2ent[idx])
    
def find_topk(name, mu, ent2idx, idx2ent, k=8):
    #ent2idx = db['ent2idx']
    #idx2ent = db['idx2ent']

    if name not in ent2idx:
        print('name not found in dictionary')
        return None

    #dist = F.cosine_similarity(mu, mu[ent2idx[name]])
    dist = torch.linalg.norm(mu - mu[ent2idx[name]], dim=1)
    for k in dist.topk(k+1, largest=False).indices.tolist():
        print(idx2ent[k])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-init-mode', type=str, default='random',
                        choices=['degree', 'random', 'constant', 'onehot'],
                        help='node feature augmentation')
    parser.add_argument('--num-features', type=int, default=64,
                        help='Feature dimensionality')
    parser.add_argument('--graph-latent-dim', type=int, default=128,
                        help='Graph hidden layer dimensionality')
    parser.add_argument('--node-latent-dim', type=int, default=128,
                        help='Encoded node latent dimensionality')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of layers in encoder/decoder')
    parser.add_argument('--edge-metric', type=str, default='freq',
                        choices=['freq', 'nmi'],
                        help='edge weight metric for graph building. ' +\
                        'Default: freq')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for the GCN layers')
    parser.add_argument('--use-batch-norm',
                        action=argparse.BooleanOptionalAction,
                        help='Use batch normalization for all layers')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Numer of epochs to train')
    
    parser.add_argument('--use-cuda', default = True, action='store_true')

    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()

    print('Configuration')
    for k, v in vars(args).items():
        print(f'- {k}={v}')


    # Data
    pm_ds = DBLPDataset(args)
    data = pm_ds.data
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)


    # GPU
    if args.use_cuda and torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    print(f'Running on {dev}')
    device = torch.device(dev)

    # Model
    model = GraphModel(args, data.edge_index, data.x.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20
    )

    for epoch in range(args.num_epochs):
        model = model.to(device)
        model.train()
        optimizer.zero_grad()

        a_pred = model(data.x.to(device), data.edge_index.to(device))
        loss = model.loss(a_pred)

        # loss = model.loss(T_data.x, T_data.train_pos_edge_index, edge_index)
        loss.backward()
        optimizer.step()

        # train loss
        if epoch % 10 == 0:
            print('epoch {}, train loss {:.3f}'.format(epoch, loss))

    model.eval()
    mu = model.get_mu(data.x, data.edge_index)
    k = 10
    find_topk('computer vision', mu, pm_ds.db['ent2idx'], pm_ds.db['idx2ent'])
    code.interact(local=dict(locals(), **globals()))