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
            'pmid': docid,
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
            del(entries[d.id]['text'])

    return entries


if __name__ == '__main__':
    n = 3000   # Total number of docs to sample
    pubmed_dir = 'data/'
    data_out_json = 'sample.jsonl'

    # Sample documents and read keywords and mesh entities from samples.
    files = [glob.glob(os.path.join(pubmed_dir, '*.gz'))]
    files = sorted([f for sublist in files for f in sublist])

    doc_cnt = 0
    bins = len(files)-1
    
    sample_ids = []
    with open(data_out_json, 'w') as fh:
        for f in files[:-1]:
            entries = sample_docs(f, n=round(n/bins)+1)
            for _, v in entries.items():
                fh.write(json.dumps(v))
                fh.write('\n')
        
