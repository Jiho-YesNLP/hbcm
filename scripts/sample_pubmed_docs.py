#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import code
import gzip
import random
import json

from lxml import etree
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.textanalytics import TextAnalyticsClient

# import constants as cfg


# credential = AzureKeyCredential(cfg.apikey)
# ta_client = TextAnalyticsClient(endpoint=cfg.endpoint,
#                                 credential=credential)


def sample_docs(f, max_n=300):
    """
    (1) Read documents that contain keywords, (2) extract the same number of
    concepts from two concept types (i.e., mesh and keywords)

    f: filepath to open
    n: number of docs to sample from the file
    """

    print(f'Processing {f}...\r', end='')
    mesh_path = 'MeshHeadingList/MeshHeading/DescriptorName/@UI'
    kw_path = './/Keyword/text()'
    # select documents that have both mesh AND keywords
    ids_ = []
    with gzip.open(f) as fh:
        root = etree.parse(fh)
        for cit in root.iter('MedlineCitation'):
            docid = cit.xpath('PMID/text()')[0]
            if len(cit.xpath(mesh_path)) > 0 and len(cit.xpath(kw_path)) > 0:
                ids_.append(docid)
    samples = random.sample(ids_, k=min(len(ids_), max_n))
    # iterate again to collect keywords, meshes, and bodoy text
    entries = dict()
    for cit in root.iter('MedlineCitation'):
        docid = cit.xpath('PMID/text()')[0]
        if docid not in samples:
            continue
        mesh_doc = cit.xpath(mesh_path)
        kw_doc = [kw.lower() for kw in cit.xpath(kw_path)]
        # try:
        #     title = cit.xpath('Article/ArticleTitle/text()')[0]
        # except IndexError:
        #     title = ''
        # try:
        #     body = cit.xpath('Article/Abstract/AbstractText/text()')[0]
        # except IndexError:
        #     body = ''
        entries[docid] = {
            'pmid': docid,
            'mesh': mesh_doc,
            'kw': kw_doc,
            # 'text': title + ' ' + body
        }
    # # run Azure key phrase extractor
    # # maximum documents allowed for KeyPhrase service is 10
    # def chunks (lst, n=10):
    #     for i in range(0, len(lst), n):
    #         yield lst[i:i+n]
            
    # for ids_ in chunks(samples):
    #     print(f'Azure KPE on {ids_[0]}...\r', end='')
    #     documents = [{
    #         'id': i,
    #         'text': entries[i]['text']
    #     } for i in ids_]

    #     response = ta_client.extract_key_phrases(documents, language='en')
    #     for d in response:
    #         entries[d.id]['kw'].extend(d.key_phrases) # allow duplicates
    #         del(entries[d.id]['text'])

    return entries


if __name__ == '__main__':
    n = 500000   # Total number of docs to sample
    pubmed_dir = 'data/pubmed/'
    data_json = 'data/sample.jsonl'

    # Sample documents and read keywords and mesh entities from samples.
    files = [glob.glob(os.path.join(pubmed_dir, '*.gz'))]
    files = sorted([f for sublist in files for f in sublist])

    doc_cnt = 0
    bins = len(files)-1
    
    with open(data_json, 'w') as fh:
        for f in files[:-1]:
            entries = sample_docs(f, max_n=round(n/bins)+1)
            for _, v in entries.items():
                fh.write(json.dumps(v))
                fh.write('\n')
        
