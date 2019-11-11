import json
import os
from random import shuffle

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _eval(x):
    try:
        return eval(x)
    except:
        return {}


out_path = "/media/ml/data_ml/image_search/images"

in_txt = "/media/ml/data_ml/image_search/meta_Clothing_Shoes_and_Jewelry.json"

with open(in_txt, 'r') as f:
    data = f.read().split("\n")

shuffle(data)

# list_q = ['Dresses', 'Tops', 'Tees', 'Shirts']
#
# data = [x for x in data if any([a in x for a in list_q])]

out_json = []

for x in tqdm(data):
    x = _eval(x)
    if "title" in x:
        out_json.append((out_path + "/" + x['asin'] + ".jpg", x['title'] + " " + x.get("description", "")))
    elif x:
        x['title'] = " ".join([a for b in x['categories'] for a in b])
        out_json.append((out_path + "/" + x['asin'] + ".jpg", x['title'] + " " + x.get("description", "")))

out_json = [x for x in out_json if os.path.exists(x[0]) and x[1].strip()]

train, test = train_test_split(out_json, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)

json.dump(train, open('../input/filtred_train_data.json', 'w'), indent=4)
json.dump(test, open('../input/filtred_test_data.json', 'w'), indent=4)
json.dump(val, open('../input/filtred_val_data.json', 'w'), indent=4)
