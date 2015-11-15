# -*- coding: utf-8 -*-
__author__ = 'catherine'

import os
import logging

logger = logging.getLogger(__name__)

import pickle
import codecs
from sklearn.utils import check_random_state
import numpy as np


class WikiSampleLoader:

    # sample file name (expected a binary pkz file)
    # the sample data should contain training and test datasets
    _file_name = os.environ['HOME']+"/scikit_learn_data/wiki_sample.pkz"

    # subset to load ('train' or 'test' or 'all'
    _subset = 'all'

    # shuffle the data y/n
    _shuffle = True

    # numpy random number generator or seed integer
    # Used to shuffle the dataset.
    _random_state = 42

    #
    # Constructor
    #
    def __init__(self, **kwargs):
        if 'file_name' in kwargs:
            self._file_name = kwargs['file_name']
        if 'subset' in kwargs:
            self._subset = kwargs['subset']
        if 'shuffle' in kwargs:
            self._shuffle = kwargs['shuffle']
        if 'random_state' in kwargs:
            self._random_state = kwargs['random_state']

    #
    # Load dataset and returns related cache data
    #
    def load_dataset(self):
        data = None
        if os.path.exists(self._file_name):
            try:
                with open(self._file_name, 'rb') as f:
                    compressed_content = f.read()
                uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
                cache = pickle.loads(uncompressed_content)
                if self._subset in ('train', 'test'):
                    data = cache[self._subset]
                elif self._subset == 'all':
                    data_lst = list()
                    target = list()
                    filenames = list()
                    for subset in ('train', 'test'):
                        data = cache[subset]
                        data_lst.extend(data.data)
                        target.extend(data.target)
                        filenames.extend(data.filenames)

                    data.data = data_lst
                    data.target = np.array(target)
                    data.filenames = np.array(filenames)
                if self._shuffle:
                    random_state = check_random_state(self._random_state)
                    indices = np.arange(data.target.shape[0])
                    random_state.shuffle(indices)
                    data.filenames = data.filenames[indices]
                    data.target = data.target[indices]
                    # Use an object array to shuffle: avoids memory copy
                    data_lst = np.array(data.data, dtype=object)
                    data_lst = data_lst[indices]
                    data.data = data_lst.tolist()
                else:
                    raise ValueError(
                        "subset can only be 'train', 'test' or 'all', got '%s'" % self._subset)

                data.description = 'wikipedia corpus dataset'
            except Exception as e:
                print(80 * '_')
                print('Cache loading failed')
                print(80 * '_')
                print(e)
        return data

"""
wsl = WikiSampleLoader()
data = wsl.load_dataset()
#print(data)
print("%d documents" % len(data.data))
print("%d categories" % len(data.target_names))
print
"""