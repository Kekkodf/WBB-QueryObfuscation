import os
from embeddings import glove
import argparse
import importlib
from utils.Timer import Timer
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import normalize

basePath = "/ssd/data/faggioli/24-ECIR-FF"
dataPath = f"{basePath}/data"
savePath = f"{dataPath}/protected-queries"

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epsilon", default=5, type=int)  # "convdr"
parser.add_argument('-m', '--mechanism', default="MahalanobisMechanism")
parser.add_argument("-n", "--ntests", default=10, type=int)
parser.add_argument('-r', '--repetitions', default=100, type=int)
parser.add_argument('-v', '--vectors', default="glove.6B.50d")

args = parser.parse_args()

representation = glove(f"{dataPath}/vectors/{args.vectors}.txt", vocab="/ssd/data/faggioli/24-ECIR-FF/data/vocabularies/msmarco-passage-vocab.txt")

m = representation.get_embeddings_matrix()
m = normalize(m)

w = normalize(representation.encode("medicine").reshape(1, -1))

coss = np.dot(w, m.T)

#np.sum(coss>=0.5) to set level treshold