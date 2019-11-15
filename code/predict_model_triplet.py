import json

import numpy as np
import pandas as pd

from model_triplet import tokenize, map_sentences, cap_sequences, read_img, image_model, text_model
from tqdm import tqdm

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":

    out_name = "../output/test_representations.json"

    mapping = json.load(open('mapping.json', 'r'))

    test = json.load(open("../input/filtred_test_data.json", 'r'))

    list_images_test, _captions_test = list(zip(*test))
    captions_test = [tokenize(x) for x in _captions_test]
    captions_test = map_sentences(captions_test, mapping)
    captions_test = cap_sequences(captions_test, 70, 0)

    file_path = "model_triplet.h5"

    t_model = text_model(vocab_size=len(mapping) + 1)
    i_model = image_model()

    t_model.load_weights(file_path, by_name=True)
    i_model.load_weights(file_path, by_name=True)

    # target_image_encoding = []
    #
    # for img_paths in tqdm(chunker(list_images_test, 128), total=len(list_images_test)//128):
    #     images = np.array([read_img(file_path) for file_path in img_paths])
    #     e = i_model.predict(images)
    #     target_image_encoding += e.tolist()
    #
    # target_text_encoding = t_model.predict(np.array(captions_test), verbose=1, batch_size=128)
    #
    # target_text_encoding = target_text_encoding.tolist()
    #
    # df = pd.DataFrame({"images": list_images_test, "text": _captions_test, "image_repr": target_image_encoding,
    #                    "text_repr": target_text_encoding})
    #
    # df.to_json(out_name, orient='records')
    #
    # data = json.load(open(out_name, 'r'))
    # json.dump(data, open(out_name, 'w'), indent=4)
    #
    # # New queries
    #
    # out_name = "../output/queries_representations.json"

    _captions_test = ['blue tshirt', 'blue shirt', 'red dress', 'halloween outfit', 'baggy jeans', 'ring',
                      'Black trousers', 'heart Pendant']

    captions_test = [tokenize(x) for x in _captions_test]
    captions_test = map_sentences(captions_test, mapping)
    captions_test = cap_sequences(captions_test, 70, 0)

    target_text_encoding = t_model.predict(np.array(captions_test), verbose=1, batch_size=128)

    target_text_encoding = target_text_encoding.tolist()

    df = pd.DataFrame({"text": _captions_test,
                       "text_repr": target_text_encoding})

    df.to_json(out_name, orient='records')

    data = json.load(open(out_name, 'r'))
    json.dump(data, open(out_name, 'w'), indent=4)



