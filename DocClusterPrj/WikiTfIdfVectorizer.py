from __future__ import print_function
# -*- coding: utf-8 -*-
__author__ = 'catherine'

from WikiSampleLoader import WikiSampleLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from time import time

class WikiTfIdfVectorizer:

    # number of features by vector
    _n_features = 15000
    # stop words (if no english should be filled with a list of words)
    _stop_words = 'english'
    # negative coordinates allowed y/n
    _non_negative = True
    # with normalization ('l1' or 'l2' or None ==> no normalization)
    _norm = None
    # True when the output matrix should be filled by 0 and 1 only
    _binary=False
    # low threshold frequency (word with lowest frequency are not taken in account)
    _min_df = 0.1
    # high threshold frequency (word with highest frequency are not taken in account)
    _max_df = 2.
    # use idf or not
    _use_idf = True
    # use hashing or not
    _use_hashing = False
    # target matrix
    _X = None
    # vectorizer
    _vectorizer = None
    # cluster list
    _cluster_list = None
    # labels
    _labels = None

    #
    # Constructor : set instance variables
    #
    def __init__(self, **kwargs):
        if 'n_features' in kwargs:
            self._n_features = kwargs['n_features']
        if 'stop_words' in kwargs:
            self._stop_words = kwargs['stop_words']
        if 'non_negative' in kwargs:
            self._non_negative = kwargs['non_negative']
        if 'norm' in kwargs:
            self._norm = kwargs['norm']
        if 'binary' in kwargs:
            self._binary = kwargs['binary']
        if 'min_df' in kwargs:
            self._min_df = kwargs['min_df']
        if 'max_df' in kwargs:
            self._max_df = kwargs['max_df']
        if 'min_df' in kwargs:
            self._min_df = kwargs['min_df']
        if 'use_idf' in kwargs:
            self._use_idf = kwargs['use_idf']
        if 'use_hashing' in kwargs:
            self._use_hashing = kwargs['use_hashing']

    def vectorize(self, wsl):
        print("loading wiki documents dataset")
        #wsl = WikiSampleLoader()
        data = wsl.load_dataset()
        self._cluster_list = data.target_names
        self._labels = data.target
        print("%d documents" % len(data.data))
        print("%d categories" % len(data.target_names))
        print
        print("Extracting features from the training dataset using a sparse vectorizer")
        t0 = time()
        if self._use_hashing:
            if self._use_idf:
                # Perform an IDF normalization on the output of HashingVectorizer
                hasher = HashingVectorizer(n_features=self._n_features,
                                           stop_words=self._stop_words,
                                           non_negative=self._non_negative,
                                           norm=self._norm, binary=self._binary)
                vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                vectorizer = HashingVectorizer(n_features=self._n_features,
                                               stop_words=self._stop_words,
                                               non_negative=self._non_negative,
                                               norm='l2',
                                               binary=self._binary)
        else:
            vectorizer = TfidfVectorizer(max_df=self._max_df,
                                         max_features=self._n_features,
                                         min_df=self._min_df,
                                         stop_words=self._stop_words,
                                         use_idf=self._use_idf)
        self._X = vectorizer.fit_transform(data.data)
        self._vectorizer = vectorizer

        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % self._X.shape)
        print()


"""
w_tf_idf = WikiTfIdfVectorizer(use_hashing=True)
w_tf_idf.vectorize()
X = w_tf_idf._X

print (X[:, 0])
"""