import json
from random import shuffle
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.neighbors import NearestNeighbors

repr_json = "../output/test_representations.json"

data = pd.read_json(repr_json)

data = data.sample(n=1000)

img_repr = data['image_repr'].tolist()
text_repr = data['text_repr'].tolist()

nn = NearestNeighbors(n_jobs=-1, n_neighbors=1000)

nn.fit(text_repr)

preds = nn.kneighbors(img_repr, return_distance=False).tolist()
ranks = []

for i, x in enumerate(preds):
    rank = x.index(i)+1
    ranks.append(rank)

print("Average rank :", np.mean(ranks))