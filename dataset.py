# Sourced from https://github.com/diode-dataset/diode-devkit/blob/master/diode.py
# I'm not sure if we're allowed to include this, but I'll leave it here for now

import os.path as osp
from itertools import chain
import json
import cv2

from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

_VALID_SPLITS = ('train', 'val', 'test')
_VALID_SCENE_TYPES = ('indoors', 'outdoor')


def check_and_tuplize_tokens(tokens, valid_tokens):
    if not isinstance(tokens, (tuple, list)):
        tokens = (tokens, )
    for split in tokens:
        assert split in valid_tokens
    return tokens

def enumerate_paths(src):
    '''flatten out a nested dictionary into an iterable
    DIODE metadata is a nested dictionary;
    One could easily query a particular scene and scan, but sequentially
    enumerating files in a nested dictionary is troublesome. This function
    recursively traces out and aggregates the leaves of a tree.
    '''
    if isinstance(src, list):
        return src
    elif isinstance(src, dict):
        acc = []
        for k, v in src.items():
            _sub_paths = enumerate_paths(v)
            _sub_paths = list(map(lambda x: osp.join(k, x), _sub_paths))
            acc.append(_sub_paths)
        return list(chain.from_iterable(acc))
    else:
        raise ValueError('do not accept data type {}'.format(type(src)))

class DIODE(Dataset):
    def __init__(self, meta_fname, data_root, splits, scene_types):
        self.data_root = data_root
        self.splits = check_and_tuplize_tokens(
            splits, _VALID_SPLITS
        )
        self.scene_types = check_and_tuplize_tokens(
            scene_types, _VALID_SCENE_TYPES
        )
        with open(meta_fname, 'r') as f:
            self.meta = json.load(f)

        imgs = []
        for split in self.splits:
            for scene_type in self.scene_types:
                _curr = enumerate_paths(self.meta[split][scene_type])
                _curr = map(lambda x: osp.join(split, scene_type, x), _curr)
                imgs.extend(list(_curr))
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        im = self.imgs[index]
        im_fname = osp.join(self.data_root, '{}.png'.format(im))
        de_fname = osp.join(self.data_root, '{}_depth.npy'.format(im))
        de_mask_fname = osp.join(self.data_root, '{}_depth_mask.npy'.format(im))

        im = np.array(Image.open(osp.join(self.data_root, im_fname)))
        im = cv2.resize(im, dsize=(512, 384), interpolation=cv2.INTER_CUBIC)
        im = im.transpose((2, 0, 1)) # Put in CHW format

        de = np.load(de_fname)
        de = cv2.resize(de, dsize=(512, 384), interpolation=cv2.INTER_CUBIC)
        de = de[:,:, np.newaxis]
        de = de.transpose((2, 0, 1)) #.squeeze()
        
        de_mask = np.load(de_mask_fname)
        de_mask = cv2.resize(de_mask, dsize=(512, 384), interpolation=cv2.INTER_CUBIC)
        #de[0][de_mask > 0] = np.reciprocal(de[0][de_mask > 0]) # 350 * reciprocal of valid locations (see pg3 Wonka)
        return im, de, de_mask