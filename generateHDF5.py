import numpy as np
import h5py
import os

import multiprocessing
from PIL import Image


batch_size = 100000
image_size = 128
num_cpus = multiprocessing.cpu_count()

def process(f):
    global image_size
    im = Image.open(f).convert('RGB')
    im = im.resize((image_size, image_size),Image.BICUBIC)
    return im

prefix = '/HOME/scz0088/run/datasets/imagenet-mini/train/'
l = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))

l_train = []

for subdir in l:
    for img_name in os.listdir(subdir):
        l_train.append(os.path.join(subdir,img_name))

i = 0
imagenet = np.zeros((len(l), image_size, image_size, 3), dtype='uint8')
pool = multiprocessing.Pool(num_cpus)
while i < len(l):
    current_batch = l_train[i:i + batch_size]    
    current_res = np.array(pool.map(process, current_batch))
    imagenet[i:i + batch_size] = current_res    
    i += batch_size
    print(i, 'images')
    
# prefix = '/HOME/scz0088/run/datasets/imagenet-mini/val/'
# l_val = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))

# i = 0
# imagenet_val = np.zeros((len(l_val), image_size, image_size, 3), dtype='uint8')
# pool = multiprocessing.Pool(multiprocessing.cpu_count())

# while i < len(l_val):
#     current_batch = l_val[i:i + batch_size]    
#     current_res = np.array(pool.map(process, current_batch))
#     imagenet_val[i:i + batch_size] = current_res    
#     i += batch_size
#     print(i, 'images')
    
# with h5py.File('/HOME/scz0088/run/datasets/hdf5/imagenet_mini-128.hdf5', 'w') as f:
#     f['train'] = imagenet
#     f['val'] = imagenet_val