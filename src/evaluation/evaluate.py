import pandas as pd
import ir_datasets
from ir_measures import AP, nDCG, P, Recall, iter_calc

collections = {'robust04': 'irds:disks45/nocr/trec-robust-2004',
               'trec-covid': 'irds:beir/trec-covid',
               'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}

dataset = ir_datasets.load(collections['msmarco-passage'])
measures = [nDCG @ 10]

def compute_measure(run, qrels):
    out = pd.DataFrame(iter_calc(measures, qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out