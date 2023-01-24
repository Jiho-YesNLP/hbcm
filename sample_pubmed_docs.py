#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HERE
2-staged reading is not efficient. let's change it to get all the information at
once
"""

"""
Script for feature extraction
- 1st scan of PubMed baseline collection for PMID's
- sample 100,000 documents
- extract concepts (i.e., MeSH id's and author provided keywords)
- run Azure keyphrase extractor for the sample documents (run this with a
  user's confirmation)
"""
# import multiprocessing as mp
import glob
import os
import code
import gzip
import random
import json

from lxml import etree
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

import constants as cfg


credential = AzureKeyCredential(cfg.apikey)
ta_client = TextAnalyticsClient(endpoint=cfg.endpoint,
                                credential=credential)

PUBMED_DIR = 'data/pubmed/'
DATA_IDS = 'sample_ids.txt'
DATA_OUT_JSON = 'sample.jsonl'
# CPU_CNT = int(.5 * mp.cpu_count())

def read_doc(f):
    ids_ = []
    cnt = int(N / 1115)
    with gzip.open(f) as fh:
        root = etree.parse(fh)
        for cit in root.iter('MedlineCitation'):
            docid = cit.xpath('PMID/text()')[0]
            ids_.append(docid)
    return random.sample(ids_, k=cnt)

# def first_scan(files):
    # """
    # Scan an entire document collection and return statistical information,
    # including the full list of PMID's.
    # """

    # print('First scan for document sampling...')
    # ids_all = []
    # def cb_collect(r):
        # nonlocal ids_all
        # ids_all.extend(r)
        # print('#ids {}\r'.format(len(ids_all)), end='')

    # pool = mp.Pool(CPU_CNT)
    # for f in files:
        # pool.apply_async(read_doc, (f,), callback=cb_collect)
    # pool.close()
    # pool.join()

    # return ids_all

def sample_docs(f, n=100):
    """
    f: filepath to open
    n: number of docs to sample from the file
    """
    print(f'Processing a file: {f}...')
    # read ids
    ids_ = []
    with gzip.open(f) as fh:
        root = etree.parse(fh)
    for cit in root.iter('MedlineCitation'):
        docid = cit.xpath('PMID/text()')[0]
        ids_.append(docid)
    samples = random.sample(ids_, k=n)
    # iterate again to collect keywords, meshes, and bodoy text
    entries = dict()
    for cit in root.iter('MedlineCitation'):
        docid = cit.xpath('PMID/text()')[0]
        if docid not in samples: continue
        mesh_doc = \
            cit.xpath('MeshHeadingList/MeshHeading/DescriptorName/@UI')
        kw_doc = [kw.lower() for kw in cit.xpath('.//Keyword/text()')]
        try:
            title = cit.xpath('Article/ArticleTitle/text()')[0]
        except IndexError:
            title = ''
        try:
            body = cit.xpath('Article/Abstract/AbstractText/text()')[0]
        except IndexError:
            body = ''
        entries[docid] = {
            'mesh': mesh_doc,
            'kw': kw_doc,
            'text': title + ' ' + body
        }
    # run Azure key phrase extractor
    # maximum documents allowed for KeyPhrase service is 10
    def chunks (lst, n=10):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    for ids_ in chunks(samples):
        print(f'Azure KPE on {ids_[0]}...\r', end='')
        documents = [{
            'id': i,
            'text': entries[i]['text']
        } for i in ids_]

        response = ta_client.extract_key_phrases(documents, language='en')
        for d in response:
            entries[d.id]['kw'].extend(d.key_phrases) # allow duplicates

    code.interact(local=dict(locals(), **globals()))





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


# def second_scan(ids, files):
    # """
    # Scan the corpus again to extract concepts (i.e., mesh and keywords) and
    # fetch key phrases using MS Azure KP extractor.
    # """
    # print('Second scan for concept extraction...')
    # q = 'Do you want to extract key phrases using Azure service? (y/N): '
    # use_kp_extractor = input(q).lower().strip() == 'y'

    # entry_cnt = 0
    # def cb_add_entries(r):
        # code.interact(local=dict(locals(), **globals()))
        # nonlocal entry_cnt
        # entry_cnt += len(r)
        # with open(DATA_OUT_JSON, 'a') as fh:
            # fh.write(json.dumps(r))
        # print('{}/{}\r'.format(len(r), entry_cnt), end='')

    # # pool = mp.Pool(CPU_CNT)
    # # for f in files:
    # #     pool.apply_async(extract_concepts, (f,ids), callback=cb_add_entries)
    # # pool.close()
    # # pool.join()

    # # Use integers for IDs for speed
    # r = extract_concepts(files[0], [int(x) for x in ids])
    # cb_add_entries(r)




if __name__ == '__main__':
    n = 3000   # Total number of docs to sample

    # Sample documents and read keywords and mesh entities from samples.
    files = [glob.glob(os.path.join(PUBMED_DIR, '*.gz'))]
    files = sorted([f for sublist in files for f in sublist])

    doc_cnt = 0
    bins = len(files)-1
    for f in files[:-1]:
        jsonl = sample_docs(f, n=round(n/bins)+1)

    # if os.path.exists(DATA_IDS):
        # ids = open(DATA_IDS).read().split()
    # else:
        # ids = first_scan(files)
    # # save sampled doc ids
    # with open(DATA_IDS, 'w') as fh:
        # fh.write(' '.join(ids))

    # second_scan(ids, files)


