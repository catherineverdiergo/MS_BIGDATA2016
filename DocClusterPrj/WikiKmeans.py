# -*- coding: utf-8 -*-
__author__ = 'catherine'

from time import time
from sklearn.cluster import KMeans, MiniBatchKMeans

class WikiKmeans:

    # number of clusters (mandatory)
    _n_clusters = None
    # centroïd initialisation k-means++ or random
    _init = 'k-means++'
    # number of initialization attempts
    _n_init = 1
    # should be larger than _n_cluster
    _init_size = 1000
    # size of the mini-batches
    _batch_size = 1000
    # verbose y/n
    _verbose = True
    # maximum number of iterations for classical K-means
    _max_iter = 100
    # Minibatch k-means or classical k-means
    _mini_batch = False

    #
    # Constructor : set instance variables
    #
    def __init__(self, nb_cluster, **kwargs):
        self._n_clusters = nb_cluster
        if 'init' in kwargs:
            self._init = kwargs['init']
        if 'n_init' in kwargs:
            self._n_init = kwargs['n_init']
        if 'init_size' in kwargs:
            self._init_size = kwargs['init_size']
        if 'batch_size' in kwargs:
            self._batch_size = kwargs['batch_size']
        if 'verbose' in kwargs:
            self._verbose = kwargs['verbose']
        if 'max_iter' in kwargs:
            self._max_iter = kwargs['max_iter']
        if 'mini_batch' in kwargs:
            self._mini_batch = kwargs['mini_batch']

    #
    # Apply K-means with a given sparse vectorized dataset
    #
    def apply_K_means(self, X):
        if self._mini_batch:
            km = MiniBatchKMeans(n_clusters=self._n_clusters,
                                 init=self._init,
                                 n_init=self._n_init,
                                 init_size=self._init_size,
                                 batch_size=self._batch_size,
                                 verbose=self._verbose)
        else:
            km = KMeans(n_clusters=self._n_clusters,
                        init=self._init,
                        max_iter=self._max_iter,
                        n_init=self._n_init,
                        verbose=self._verbose)

        print("Clustering sparse data with %s" % km)
        t0 = time()
        km.fit(X)
        print("done in %0.3fs" % (time() - t0))
        print
        return km

