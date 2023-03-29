import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import pickle
import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import os.path as osp
from collections import Counter

class DBLPDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.db_file = 'edge_list_9000.pkl'
        self.processed_file = 'data_new_9000.pt'
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
        return len(self.data.x.size[0])

    def process(self):
        print(f'Processing db file ({self.db_file})')

        filtered_edges = {(u, v): s for (u, v), s in self.db['edges'].items() if s >= 4}
        edge_index = torch.tensor(list(filtered_edges.keys())).t().contiguous()

        #num_nodes = len(self.db['ent2idx'])

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

        self.data = Data(x=x, edge_index=edge_index)


# Load the saved graph and create the dataset
#data = load_graph_dataset('edge_list_5000.pkl', min_freq=3)
