import json
from collections import defaultdict, Counter
from itertools import combinations
from tqdm import tqdm
import pickle

def build_concept2index(fp):
    cpt2idx = defaultdict(lambda: len(cpt2idx))
    idx2cpt = dict()

    with open(fp) as fh:
        data = json.load(fh)
        for doc_id, doc in data.items():
            keywords = [kw.lower() for kw in doc.get('keywords', [])]
            fos = [f.lower() for f in doc.get('fos', [])]

            for cpt in keywords + fos:
                if cpt not in cpt2idx:
                    index = cpt2idx[cpt]
                    idx2cpt[index] = cpt

    return dict(cpt2idx), idx2cpt

def build_edgelist(fp, c2i):
    edges = Counter()
    node_df = Counter() # document frequency
    num_docs = 0
    
    with open(fp) as fh:
        data = json.load(fh)
        for doc_id, doc in data.items():
            num_docs += 1
            keywords = [kw.lower() for kw in doc.get('keywords', [])]
            fos = [f.lower() for f in doc.get('fos', [])]

            concepts = [c2i[c] for c in keywords + fos if c in c2i]

            edges.update(combinations(sorted(concepts), 2))
            node_df.update(concepts)

    return edges, node_df, num_docs


# Example usage:
if  __name__ == '__main__':
    input_file = 'output_entries_file_9000.json'
    db_file = 'edge_list_9000.pkl'
    edge_metric = 'freq'
    concept2index, index2concept = build_concept2index(input_file)
    #print('c2i:',concept2index)
    #print('c2i:',index2concept)
    edges, node_df, num_docs = build_edgelist(input_file, concept2index)
    print(num_docs)
    print(f'Saving a graph db in {db_file}: ')
    print(f'#edges: {len(edges)}, #nodes: {len(concept2index)}')
    db = {
      'edges': edges,
      'ent2idx': concept2index,
      'idx2ent': index2concept,
      'idx2df': node_df,
      'collection_size': num_docs
   }
    with open(db_file, 'wb') as f:
      pickle.dump(db, f)

