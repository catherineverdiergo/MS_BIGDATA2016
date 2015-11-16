from __future__ import print_function
# -*- coding: utf-8 -*-
__author__ = 'catherine'

from WikiSampleLoader import WikiSampleLoader
from WikiTfIdfVectorizer import WikiTfIdfVectorizer
from WikiKmeans import WikiKmeans
from sklearn import metrics
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import logging
from optparse import OptionParser
import os

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--dataset_file",
              type="string", dest="dataset_file",
              default=os.environ['HOME']+"/scikit_learn_data/wiki_sample.pkz",
              help="Specify dataset file (pkz format expected).")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--init",
              type="string", dest="init", default='k-means++',
              help="K-means centroid initialization (k-means++ or random).")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()

# Initialize WikiSampleLoader
wsl = WikiSampleLoader(file_name=opts.dataset_file)

# Add wiki frequent technical words to stop_words
# to avoid overfitting on not relevant terms
stop_w = set(ENGLISH_STOP_WORDS)
# Enrich stop_words set with wiki frequent technical tags
stop_w = stop_w.union(['url', 'http', 'www', 'ref', 'jpg', 'file', 'com'])
stop_w = stop_w.union(['web', 'category', 'reference', 'title', 'org', 'br'])

w_tf_idf = WikiTfIdfVectorizer(stop_words=stop_w,
                               use_idf=opts.use_idf,
                               n_features=opts.n_features,
                               use_hashing=opts.use_hashing)
w_tf_idf.vectorize(wsl)

# get vectorized dataset
X = w_tf_idf.get_vectorized_dataset()
# init K-means
k = len(w_tf_idf.get_cluster_list())
labels = w_tf_idf.get_label_vector()
wkm = WikiKmeans(k, verbose=opts.verbose, mini_batch=opts.minibatch, init=opts.init)
# apply K-means
km = wkm.apply_K_means(X)

#print(labels)
#print(km.labels_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

if not opts.use_hashing:
    print()
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = w_tf_idf.get_vectorizer().get_feature_names()
    for i in range(k):
        print("Cluster %d" % i)
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
