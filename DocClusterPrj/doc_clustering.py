# -*- coding: utf-8 -*-
__author__ = 'catherine'

from WikiTfIdfVectorizer import WikiTfIdfVectorizer
from WikiKmeans import WikiKmeans
from sklearn import metrics

# Load dataset from pkz file and vectorize it
#w_tf_idf = WikiTfIdfVectorizer(use_hashing=True)
w_tf_idf = WikiTfIdfVectorizer()
w_tf_idf.vectorize()
# get vectorized dataset
X = w_tf_idf._X
# init K-means
k = len(w_tf_idf._cluster_list)
labels = w_tf_idf._labels
wkm = WikiKmeans(k, mini_batch=True)
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

