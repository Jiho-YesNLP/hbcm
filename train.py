"""
https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery/blob/2c3145dff638d9a1e0706874dcfad6132771807d/multi_fidelity_modelling/DR_modelling/deep_learning/train_dr.py
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

    
def find_topk(db, mu, name, k=6):
    ent2idx = db['ent2idx']
    idx2ent = db['idx2ent']

    if name not in ent2idx:
        print('name not found in dictionary')
        return None

    dist = F.cosine_similarity(mu, mu[ent2idx[name]])
    for k in dist.topk(6, largest=False).indices.tolist()[1:]:
        print(idx2ent[k])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-features', type=int, default=64,
                        help='Feature dimensionality')
    parser.add_argument('--graph-latent-dim', type=int, default=64,
                        help='Graph hidden layer dimensionality')
    parser.add_argument('--node-latent-dim', type=int, default=32,
                        help='Encoded node latent dimensionality')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers in encoder/decoder')
    parser.add_argument('--use-batch-norm',
                        action=argparse.BooleanOptionalAction,
                        help='Use batch normalization for all layers')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='Numer of epochs to train')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Data
    pm_ds = PubmedDataset(args)
    edge_index = pm_ds.data.edge_index
    T_data = pyg_utils.train_test_split_edges(pm_ds.data)

    # Model
    model = GraphModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20
    )

    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(T_data.x, T_data.train_pos_edge_index, edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            model.eval()
            val_loss = model.loss(T_data.x, T_data.val_pos_edge_index, edge_index)
            scheduler.step(val_loss)  # decay learning rate

        if epoch % 10 == 0:
            model.eval()
            roc_auc, ap = model.single_test(T_data.x,
                                            T_data.train_pos_edge_index,
                                            T_data.test_pos_edge_index,
                                            T_data.test_neg_edge_index)
            print('Epoch {} (lr {}) - Loss: {:.3f} ROC_AUC: {:.3f} Precision: {:.3f}'
                  ''.format(epoch, scheduler._last_lr[0], loss.cpu().item(),
                            roc_auc, ap))
            if scheduler._last_lr[0] < args.lr * 0.1:
                break

    mu = model.get_mu(T_data.x, edge_index)
    code.interact(local=dict(locals(), **globals()))
