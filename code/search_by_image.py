import json
from random import shuffle
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from model_triplet import read_img
import cv2
from uuid import uuid4


repr_json = "../output/test_representations.json"

data = pd.read_json(repr_json)

#data = data.sample(n=50000)

img_repr = data['image_repr'].tolist()
img_paths = data['images'].tolist()
text_repr = data['text_repr'].tolist()

nn = NearestNeighbors(n_jobs=-1, n_neighbors=9)

nn.fit(img_repr)

preds = nn.kneighbors(img_repr[:100], return_distance=False).tolist()

most_similar_images = []
query_image = []


for i, x in enumerate(preds):
    preds_paths = [img_paths[i] for i in x]
    query_image.append(preds_paths[0])
    most_similar_images.append(preds_paths[1:])

for q, similar in zip(query_image, most_similar_images):
    fig, axes = plt.subplots(3, 3)
    all_images = [q]+similar

    for idx, img_path in enumerate(all_images):
        i = idx % 3  # Get subplot row
        j = idx // 3  # Get subplot column
        image = read_img(img_path, preprocess=False)
        image = image[:, :, ::-1]
        axes[i, j].imshow(image/255)
        axes[i, j].axis('off')
        axes[i, j].axis('off')
        if idx == 0:
            axes[i, j].set_title('Query Image')
        else:
            axes[i, j].set_title('Result Image %s'%idx)

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig('../output/images/%s.png'%uuid4().hex)

