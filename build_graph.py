"""
build_graph.py

This script reads the store jsonl file of concepts and construct a graph (or 
graph data for later use).
"""
import code

import json
from collections import Counter, defaultdict
from itertools import combinations
import math
import pickle

from lxml import etree
import numpy as np


def read_mesh(fp):
    print('Reading MeSH concepts..')
    meshes = []
    with open(fp) as fh:
        root = etree.parse(fh)
        for m in root.iter('DescriptorRecord'):
            mshid = m.xpath('DescriptorUI/text()')[0]
            meshes.append(mshid)
    return meshes
    
def read_kw(fp, n=30000):
    print(f'Reading {n} Keyword concepts..')
    kw_counter = Counter()
    with open(fp) as fh:
        for l in fh:
            doc = json.loads(l)
            kw_counter.update(map(str.lower, doc['kw']))
    # print stats
    vals = list(kw_counter.values())
    print('keyword frequency stats: min {}, max {}, mean {:.3f}, std {:.3f}'
          ''.format(np.min(vals), np.max(vals), np.mean(vals), np.std(vals)))
    return [kw for kw, v in kw_counter.most_common(n)]

def build_edgelist(fp, c2i):
    """
    Extract edges between concepts, count document frequencies (df) of concepts
    :param fp: filepath to the json data
    :param c2i: concept 2 index mapping
    :return: (edges, df, |C|)
    """
    edges = Counter()
    node_df = Counter() # document frequency (aobut 1/3 of all concepts appear)
    num_docs = 0
    
    # extract edges from docs
    with open(fp) as fh:
        for l in fh:
            num_docs += 1
            doc = json.loads(l)
            concepts = [c2i[c] for c in doc['kw'] + doc['mesh'] if c in c2i]
            edges.update(combinations(sorted(concepts), 2))
            node_df.update(concepts)
    
    
    # weight normalization (NMI normalized mutual information)
    for (u, v) in edges:
        p_u = node_df[u] / num_docs
        p_v = node_df[v] / num_docs
        p_uv = edges[(u, v)] / num_docs
        I = p_uv * math.log(p_uv / (p_u * p_v))
        if p_u > p_uv:
            I += (p_u - p_uv) * math.log((p_u - p_uv) / (p_u * (1 - p_v)))
        if p_v > p_uv:
            I += (p_v - p_uv) * math.log((p_v - p_uv) / (p_v * (1 - p_u)))
        I += (1 - p_u - p_v + p_uv) * math.log((1 - p_u - p_v + p_uv) / ((1 - p_u) * (1 - p_v)))
        h_u = - p_u * math.log(p_u) - (1 - p_u) * math.log(1 - p_u)
        h_v = - p_v * math.log(p_v) - (1 - p_v) * math.log(1 - p_v)
        w = 2 * I / (h_u + h_v)
        edges[(u, v)] = w
        
    return edges, node_df, num_docs
    
    
if __name__ == '__main__':
    data_json = 'data/sample.jsonl'
    mesh_file = 'data/desc2023.xml'
    db_file = 'data/raw/edgelist.pkl'
    
    # Build a vocabulary of biomedical concepts (i.e., mesh and keywords)
    meshes = read_mesh(mesh_file)
    keywords = read_kw(data_json, n=10000)
    
    # Build a vocabulary of concepts
    cpt2idx = defaultdict(lambda: len(cpt2idx))
    idx2cpt = dict()
    for cpt in meshes + keywords:
        idx2cpt[cpt2idx[cpt]] = cpt
        
    # build a graph
    edges, node_df, num_docs = build_edgelist(data_json, cpt2idx)
    
    # save
    print(f'Saving a graph db in {db_file}: ')
    print(f'#edges: {len(edges)}, #nodes: {len(cpt2idx)}')
    db = {
        'edges': edges,
        'ent2idx': dict(cpt2idx),
        'idx2ent': idx2cpt,
        'idx2df': node_df,
        'collection_size': num_docs
    }
    with open(db_file, 'wb') as f:
        pickle.dump(db, f)
        
    # code.interact(local=dict(locals(), **globals()))
