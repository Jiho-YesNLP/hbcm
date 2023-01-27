import code
import pickle
import os
import os.path as osp
from typing import Callable, List, Optional

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
    r"""Graph of biomedical concepts in Pubmed. Concepts are the union set of 
    MeSH codes, author provided keywords, and keyword phrases extracted using 
    Azure cognitive service.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    - #nodes: 
    - #edges: 
    - #features: 
    - #classes: 
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        emb_dim: int = 64
    ):
        self.emb_dim = emb_dim
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        return ['edgelist.pkl']
        
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
        
    def process(self):
        db_file = osp.join(self.raw_dir, self.raw_file_names[0])
        db = pickle.load(open(db_file, 'rb'))
        # x
        x = nn.Embedding(len(db['ent2idx']), self.emb_dim)
        # graph
        rows = []
        cols = []
        for (u, v), nmi in db['edges'].items():
            if nmi > 0.35:
                rows.extend([u, v])
                cols.extend([v, u])
        edge_index = torch.stack([torch.tensor(rows).to(torch.long), 
                                  torch.tensor(cols).to(torch.long)], dim=0)
        edge_index = coalesce(edge_index, num_nodes=len(db['ent2idx']))
        
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
        