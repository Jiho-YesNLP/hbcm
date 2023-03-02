"""
Goal: We want to discover the keywords that occur relatively frequently but
not having a graphical commonality with other MeSH codes.
For example, `text neck syndrome' which doesn't have corresponding concept in
MeSH vocabulary, but a popular term.

We also want to learn the graph topology with weighted edges. Will this be
simple as calculating BCE with real-valued adjacency matrix?
"""
import code
import pickle
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils

from model import GraphModel
from data_loading import PubmedDataset

    
def find_topk(name, mu, ent2idx, idx2ent, k=10):
    # ent2idx = db['ent2idx']
    # idx2ent = db['idx2ent']

    if name not in ent2idx:
        print('name not found in dictionary')
        return None

    # dist = F.cosine_similarity(mu, mu[ent2idx[name]])
    dist = torch.linalg.norm(mu - mu[ent2idx[name]], dim=1)
    for k in dist.topk(k+1, largest=False).indices.tolist():
        print(idx2ent[k])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-init-mode', type=str, default='degree',
                        choices=['degree', 'random', 'constant', 'onehot'],
                        help='node feature augmentation')
    parser.add_argument('--num-features', type=int, default=64,
                        help='Feature dimensionality')
    parser.add_argument('--graph-latent-dim', type=int, default=128,
                        help='Graph hidden layer dimensionality')
    parser.add_argument('--node-latent-dim', type=int, default=128,
                        help='Encoded node latent dimensionality')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of layers in encoder/decoder')
    parser.add_argument('--edge-metric', type=str, default='freq',
                        choices=['freq', 'nmi'],
                        help='edge weight metric for graph building. ' +\
                        'Default: freq')
    parser.add_argument('--use-batch-norm',
                        action=argparse.BooleanOptionalAction,
                        help='Use batch normalization for all layers')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='Numer of epochs to train')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use-cuda', action='store_true')
    args = parser.parse_args()

    print('Configuration')
    for k, v in vars(args).items():
        print(f'- {k}={v}')


    # Data
    pm_ds = PubmedDataset(args)
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
    model = GraphModel(args, data.edge_index.to(device), data.x.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.5, patience=20
    # )
    
    for epoch in range(args.num_epochs):
        model = model.to(device)
        model.train()
        optimizer.zero_grad()

        a_pred = model(data.x.to(device))
        loss = model.loss(a_pred)

        # loss = model.loss(T_data.x, T_data.train_pos_edge_index, edge_index)
        loss.backward()
        optimizer.step()

        # train loss
        if epoch % 2 == 0:
            print('epoch {}, train loss {:.3f}'.format(epoch, loss))
            # scheduler.step(val_loss)  # decay learning rate

        # if epoch % 10 == 0:
        #     model.eval()
        #     roc_auc, ap = model.single_test(data.x,
        #                                     data.train_pos_edge_index,
        #                                     data.test_pos_edge_index,
        #                                     data.test_neg_edge_index)
        #     print('Epoch {} (lr {}) - Loss: {:.3f} ROC_AUC: {:.3f} Precision: {:.3f}'
        #           ''.format(epoch, scheduler._last_lr[0], loss.cpu().item(),
        #                     roc_auc, ap))
        #     if scheduler._last_lr[0] < args.lr * 0.1:
        #         break


    model.eval()
    mu = model.get_mu(data.x)
    find_topk('cancer', mu, pm_ds.db['ent2idx'], pm_ds.db['idx2ent'])
    code.interact(local=dict(locals(), **globals()))
