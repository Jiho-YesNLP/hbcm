import code

import pickle
import os
import os.path as osp
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data


class PubmedDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.db_file = 'data/edgelist.pkl'
        self.processed_file = 'data/data.pt'
        self.data = None
        self.db = pickle.load(open(osp.join(self.db_file), 'rb'))

        # setup num_features
        if args.x_init_mode == 'degree':
            args.num_features = 2
        elif args.x_init_mode == 'constant':
            args.num_features = 1
        elif args.x_init_mode == 'onehot':
            args.num_features = len(self.db['idx2ent'])
        self.process()

    def __len__(self):
        return len(self.data.x.size(0))

    def process(self):
        print(f'Processing db file ({self.db_file})')
        # graph
        e_from = []
        e_to = []
        for (u, v), s in self.db['edges'].items():
            if self.args.edge_metric == 'nmi':
                if s < 0.35:
                    continue
            if self.args.edge_metric == 'freq':
                if s < 3:
                    continue
            e_from.extend([u, v])
            e_to.extend([v, u])

        # Input initialization
        if self.args.x_init_mode == 'random':
            x = torch.randn(len(self.db['ent2idx']), self.args.num_features)
        elif self.args.x_init_mode == 'degree':
            degree_cnt = Counter()
            degree_cnt.update([u for (u, v) in self.db['edges']])
            x = torch.zeros(len(self.db['ent2idx']), self.args.num_features)  # 2 for df and degree
            for i in range(len(self.db['idx2ent'])):
                x[i][0] = self.db['idx2df'][0]
                x[i][1] = degree_cnt[0]
        elif self.args.x_init_mode == 'onehot':
            x = F.one_hot(torch.arange(self.args.num_features)).float()
        elif self.args.x_init_mode == 'constant':
            x = torch.ones(len(self.db['ent2idx']), 1)

        edge_index = torch.stack([torch.tensor(e_from).to(torch.long),
                                  torch.tensor(e_to).to(torch.long)], dim=0)

        self.data = Data(x=x, edge_index=edge_index)

