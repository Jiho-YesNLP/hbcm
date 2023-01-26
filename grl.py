import torch
from torch_geometric.data import Data


data = Data(x=x, edge_index=edge_index)


if __name__ == '__main__':
    edgelist_fp = 'data/edgelist.pkl'