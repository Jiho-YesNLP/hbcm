#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HERE
2-staged reading is not efficient. let's change it to get all the information at once
"""

"""
Script for feature extraction
- 1st scan of PubMed baseline collection for PMID's
- sample 100,000 documents
- extract concepts (i.e., MeSH id's and author provided keywords)
- run Azure keyphrase extractor for the sample documents (run this with a
  user's confirmation)
"""
import multiprocessing as mp
import glob
import os
import code
import gzip
import random
import json
from lxml import etree

PUBMED_DIR = '/home/data/pubmed22/'
DATA_IDS = 'sample_ids.txt'
DATA_OUT_JSON = 'sample.jsonl'
CPU_CNT = int(.5 * mp.cpu_count())
N = 300000   # Number of docs to sample

def read_doc(f):
    ids_ = []
    cnt = int(N / 1115)
    with gzip.open(f) as fh:
        root = etree.parse(fh)
        for cit in root.iter('MedlineCitation'):
            docid = cit.xpath('PMID/text()')[0]
            ids_.append(docid)
    return random.sample(ids_, k=cnt)

def first_scan(files):
    """
    Scan an entire document collection and return statistical information,
    including the full list of PMID's.
    """

    print('First scan for document sampling...')
    ids_all = []
    def cb_collect(r):
        nonlocal ids_all
        ids_all.extend(r)
        print('#ids {}\r'.format(len(ids_all)), end='')

    pool = mp.Pool(CPU_CNT)
    for f in files:
        pool.apply_async(read_doc, (f,), callback=cb_collect)
    pool.close()
    pool.join()

    return ids_all


def extract_concepts(f, ids):
    sample_entries = []
    with gzip.open(f) as fh:
        root = etree.parse(fh)
        for cit in root.iter('MedlineCitation'):
            docid = cit.xpath('PMID/text()')[0]
            if int(docid) not in ids: continue
            mesh_doc = \
                cit.xpath('MeshHeadingList/MeshHeading/DescriptorName/@UI')
            kw_doc = \
                cit.xpath('KeywordList/Keyword/text()')
            kw_doc = list(map(lambda x: ' '.join(x.lower().split()), kw_doc))
            sample_entries.append({
                'pmid': docid,
                'mesh': mesh_doc,
                'kw': kw_doc
            })
    return sample_entries


def second_scan(ids, files):
    """
    Scan the corpus again to extract concepts (i.e., mesh and keywords) and
    fetch key phrases using MS Azure KP extractor.
    """
    print('Second scan for concept extraction...')
    q = 'Do you want to extract key phrases using Azure service? (y/N): '
    use_kp_extractor = input(q).lower().strip() == 'y'

    entry_cnt = 0
    def cb_add_entries(r):
        code.interact(local=dict(locals(), **globals()))
        nonlocal entry_cnt
        entry_cnt += len(r)
        with open(DATA_OUT_JSON, 'a') as fh:
            fh.write(json.dumps(r))
        print('{}/{}\r'.format(len(r), entry_cnt), end='')

    # pool = mp.Pool(CPU_CNT)
    # for f in files:
    #     pool.apply_async(extract_concepts, (f,ids), callback=cb_add_entries)
    # pool.close()
    # pool.join()

    # Use integers for IDs for speed
    r = extract_concepts(files[0], [int(x) for x in ids])
    cb_add_entries(r)




if __name__ == '__main__':
    files = [glob.glob(os.path.join(PUBMED_DIR, '*.gz'))]
    files = sorted([f for sublist in files for f in sublist])

    if os.path.exists(DATA_IDS):
        ids = open(DATA_IDS).read().split()
    else:
        ids = first_scan(files)
    # save sampled doc ids
    with open(DATA_IDS, 'w') as fh:
        fh.write(' '.join(ids))

    second_scan(ids, files)


