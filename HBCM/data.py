import code
import pickle
import os
import os.path as osp
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import coalesce


class PubmedDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.n2m = {}  # Index mapping from original to the ones on graph
        self.m2n = None  # Inverse of n2m
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['edgelist.pkl']
        
    @property
    def processed_file_names:
        return ['data.pt']
        
    def process(self):
        db_file = osp.join(self.raw_dir, self.raw_file_names[0])
        db = pickle.load(open(db_file, 'rb'))
        # Read nodes and their features (df); read only the ones that have  at
        # least one edge associated with

        # graph
        rows = []
        cols = []
        connected = Counter()
        for (u, v), nmi in db['edges'].items():
            if nmi > 0.35:
                rows.extend([u, v])
                cols.extend([v, u])
                connected.update([u, v])

        # x: node features
        self.m2n = sorted(connected.keys())
        for m, n in enumerate(self.m2n):
            self.n2m[n] = m
        x = torch.zeros((len(self.m2n), 1))
        for n, df in db['idx2df'].items():
            if n in self.n2m:
                x[self.n2m[n]] = df
        x = (x - x.mean(axis=0)) / (x.std(axis=0))  # standard scale

        # index mapping (n -> m)
        rows = [self.n2m[n] for n in rows]
        cols = [self.n2m[n] for n in cols]

        edge_index = torch.stack([torch.tensor(rows).to(torch.long),
                                  torch.tensor(cols).to(torch.long)], dim=0)
        edge_index = coalesce(edge_index, num_nodes=len(connected))
        
        data = Data(x=x, edge_index=edge_index)
        torch.save(self.collate([data]), self.processed_paths[0])
        
        
        
        # db = pickle.load(open(fp, 'rb'))
        # rows = []
        # cols = []
        # # keys ['edges', 'ent2idx', 'idx2ent', 'idx2df', 'collection_size']
        
        # # note. I am not sure if using weight graph with GCN is theoretically 
        # # correct. Here, I will test it with unweight graph where edge is 
        # # defined when NMI > 0.35 (this is near the mean)
        # for (u, v), nmi in db['edges'].items():
        #     if nmi > 0.35:
        #         rows.extend([u, v])
        #         cols.extend([v, u])
        # edge_index = torch.tensor([rows, cols])
        # x = nn.Embedding(len(db['ent2idx']), self.embedding_dim)
        # data = Data(x=x, edge_index=edge_index)
        # data.num_nodes = len(db['ent2idx'])
        # self.data, self.slices = self.collate([data])
        
