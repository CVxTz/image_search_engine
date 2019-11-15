import json
from random import shuffle
import pandas as pd
import numpy as np
from matplotlib import pyplot


out_name = "../output/test_representations.json"

data = pd.read_json(out_name)

data = data.sample(n=10000)

img_repr = data['image_repr'].tolist()
img_repr_random = data['image_repr'].tolist()
shuffle(img_repr_random)
text_repr = data['text_repr'].tolist()

target_distances = []
random_distances = []

for img, random_image, text in zip(img_repr, img_repr_random, text_repr):
    d_1 = np.linalg.norm(np.array(img)-np.array(text))
    d_2 = np.linalg.norm(np.array(random_image)-np.array(text))

    target_distances.append(d_1)
    random_distances.append(d_2)


pyplot.hist(target_distances, bins=100, alpha=0.5, label='matched text')
pyplot.hist(random_distances, bins=100, alpha=0.5, label='random text')
pyplot.legend(loc='upper right')
pyplot.show()