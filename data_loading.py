import code

import pickle
import os
import os.path as osp
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data


class PubmedDataset(Dataset):
    def __init__(self, args):
        self.db_file = 'data/edgelist.pkl'
        self.processed_file = 'data/data.pt'
        self.emb_dim = args.num_features
        self.data = None
        self.db = pickle.load(open(osp.join(self.db_file), 'rb'))
        if os.path.exists(self.processed_file):
            self.data = torch.load(self.processed_file)
        else:
            self.process()

    def __len__(self):
        return len(self.data.x.size(0))

    def process(self):
        print(f'Processing db file ({self.db_file})')
        # graph
        rows = []
        cols = []
        connected = Counter()
        for (u, v), nmi in self.db['edges'].items():
            if nmi > 0.35:
                rows.extend([u, v])
                cols.extend([v, u])
                connected.update([u, v])

        self.m2n = sorted(connected.keys())
        self.n2m = dict()
        for m, n in enumerate(self.m2n):
            self.n2m[n] = m

        # x: node features (random initial embeddings)
        # x = nn.Embedding(len(m2n), self.emb_dim)
        x = torch.randn(len(self.db['ent2idx']), self.emb_dim)

        # # index mapping (n -> m)
        # rows = [n2m[n] for n in rows]
        # cols = [n2m[n] for n in cols]

        edge_index = torch.stack([torch.tensor(rows).to(torch.long),
                                  torch.tensor(cols).to(torch.long)], dim=0)
        # edge_index = pyg_utils.coalesce(edge_index, num_nodes=len(connected))

        self.data = Data(x=x, edge_index=edge_index)
        torch.save(self.data, self.processed_file)
