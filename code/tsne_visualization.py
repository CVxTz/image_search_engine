import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import manifold

out_name = "../output/test_representations.json"

data = pd.read_json(out_name)

n = 10000

data = data.sample(n=n)

img_repr = data['image_repr'].tolist()
text_repr = data['text_repr'].tolist()

tsne = manifold.TSNE(n_components=2, init='random',
                     random_state=0)

X = tsne.fit_transform(img_repr + text_repr).tolist()

img_repr_2d = X[:n]
text_repr_2d = X[n:]

distances = []
for a, b in zip(img_repr_2d, text_repr_2d):
    distances.append(np.fabs(a[0] - b[0]) + np.fabs(a[1] - b[1]))

quantile = np.quantile(distances, q=0.9)

fig = plt.figure()
plt.scatter([a[0] for a in img_repr_2d], [a[1] for a in img_repr_2d], c="r", s=4, alpha=0.5, label="Images")
plt.scatter([a[0] for a in text_repr_2d], [a[1] for a in text_repr_2d], c="b", s=4, alpha=0.5, label="Texts")
for a, b in zip(img_repr_2d, text_repr_2d):
    if np.fabs(a[0] - b[0]) + np.fabs(a[1] - b[1]) < quantile:
        plt.plot([a[0], b[0]], [a[1], b[1]], c="g", lw=0.2)
plt.legend()
plt.title('TSNE Visualization')
plt.show()
