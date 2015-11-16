# -*- coding: utf-8 -*-
__author__ = 'catherine'

from WikiSampleLoader import WikiSampleLoader
from WikiTfIdfVectorizer import WikiTfIdfVectorizer
from WikiKmeans import WikiKmeans
from sklearn import metrics
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import os

# Load dataset from pkz file and vectorize it
#w_tf_idf = WikiTfIdfVectorizer(use_hashing=True)
wsl = WikiSampleLoader()
#wsl = WikiSampleLoader(file_name=os.environ['HOME']+"/scikit_learn_data/20news-bydate.pkz")
stop_w = set(ENGLISH_STOP_WORDS)
stop_w = stop_w.union(['url', 'http', 'www', 'ref', 'jpg', 'file', 'com'])
stop_w = stop_w.union(['web', 'category', 'reference', 'title', 'org', 'br'])
w_tf_idf = WikiTfIdfVectorizer(stop_words=stop_w)
w_tf_idf.vectorize(wsl)
# get vectorized dataset
X = w_tf_idf._X
# init K-means
k = len(w_tf_idf._cluster_list)
labels = w_tf_idf._labels
wkm = WikiKmeans(k)
# apply K-means
km = wkm.apply_K_means(X)

print(labels)
print(km.labels_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = w_tf_idf._vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d: %s" % (i, w_tf_idf._cluster_list[i]))
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print
