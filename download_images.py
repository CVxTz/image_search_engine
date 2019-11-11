import multiprocessing
import os
from io import BytesIO
from random import shuffle
from urllib import request

from PIL import Image
from tqdm import tqdm


hdr = {'User-Agent': 'Mozilla/5.0'}


def download_image(x):
    filename, url = x
    try:
        if os.path.exists(filename):
            return 0
        req = request.Request(url, headers=hdr)
        response = request.urlopen(req, timeout=10)
        image_data = response.read()

        pil_image = Image.open(BytesIO(image_data))

        pil_image_rgb = pil_image.convert('RGB')

        pil_image_rgb.save(filename, format='JPEG', quality=90)
        return 0
    except:
        print("Failed : %s "%filename)
        # pil_image = Image.fromarray(np.zeros((512, 256, 3)).astype('uint8'))
        #
        # pil_image_rgb = pil_image.convert('RGB')
        #
        # pil_image_rgb.save(filename, format='JPEG', quality=90)
        return 1


out_path = "/media/ml/data_ml/image_search/images"

in_txt = "/media/ml/data_ml/image_search/meta_Clothing_Shoes_and_Jewelry.json"

with open(in_txt, 'r') as f:
    data = f.read().split("\n")

shuffle(data)

#list_q = ['Dresses', 'Tops', 'Tees', 'Shirts']

data = (eval(x) for x in data) #if any([a in x for a in list_q])

os.makedirs(out_path, exist_ok=True)


data = ((out_path+"/"+x['asin']+".jpg", x.get('imUrl', "")) for x in data)

pool = multiprocessing.Pool(processes=20)  # Num of CPUs

failures = sum(tqdm(pool.imap_unordered(download_image, data)))
print('Total number of download failures:', failures)
pool.close()
pool.terminate()
