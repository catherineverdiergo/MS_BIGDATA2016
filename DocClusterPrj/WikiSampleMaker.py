# -*- coding: utf-8 -*-
__author__ = 'catherine'

"""
    This class is dedicated to create a sample of documents from wikipedia
    in order to be used in document clustering process
"""

import requests
# import json
import numpy as np
import urllib
import os
import uuid
import tarfile
import logging

logger = logging.getLogger(__name__)

import sklearn.datasets.base as base
import pickle
import codecs
import re
import shutil

class WikiSampleMaker:
    # instance variables
    _min_doc_per_class = 5  # minimum documents number per class
    _max_doc_per_class = 20  # maximum documents number per class
    _category_request_limit = 200  # number of documents asked for each api list request
    _locale = 'en'  # default locale
    _target_dir = None
    _nb_doc_per_topic = None

    # by default, the topic list is the following:
    _topic_list = ['History', 'Literature', 'Medicine', 'Music', 'Painting', 'Physics']

    def __init__(self, **kwargs):
        if 'topic_list' in kwargs:
            self._topic_list = kwargs['topic_list']
        if 'cmlimit' in kwargs:
            self._category_request_limit = kwargs['cmlimit']
        if 'locale' in kwargs:
            self._locale = kwargs['locale']
        if 'target_dir' in kwargs:
            self._target_dir = kwargs['target_dir']
        else:
            self._target_dir = os.environ['HOME']
        if 'nb_doc' in kwargs:
            self._nb_doc_per_topic = kwargs['nb_doc']

    def build_category_url(self, topic):
        url = "https://" + self._locale
        url += ".wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:"
        url += topic + "&cmlimit=" + str(self._category_request_limit) + "&format=json"
        return url

    def build_document_url(self, title):
        title = title.encode('utf-8')
        title = urllib.quote_plus(title)
        title = urllib.quote(title)
        title = urllib.unquote_plus(title)
        url = "https://" + self._locale + ".wikipedia.org/w/api.php?action=query&titles="
        url += title.replace(" ", "_") + "&prop=revisions&rvprop=content&format=json"
        url.replace("'", "''")
        return url

    #
    # Find a set of wikipedia documents related with the given topic
    # returns a list of documents urls
    #
    def get_doc_sample(self, topic, result, nb_docs):
        if result is None:
            # result = {"topic": '"' + topic + '"', "documents": []}
            result = []
        url = self.build_category_url(topic)
        request = requests.get(url)
        if request.status_code == 200:
            doc_list = request.json()
            if nb_docs is None:
                nb_docs = np.random.randint(self._min_doc_per_class, self._max_doc_per_class)
            for idoc in range(0, nb_docs):
                rank = np.random.randint(0,
                                         min(len(doc_list["query"]["categorymembers"]),
                                             self._category_request_limit))
                selected_doc = doc_list["query"]["categorymembers"][rank]
                if not selected_doc["title"][0:9] == "Category:":
                    # result["documents"].append(self.build_document_url(selected_doc["title"]))
                    result.append(self.build_document_url(selected_doc["title"]))
                else:
                    print ("lookup for category:" + selected_doc["title"])
                    result = self.get_doc_sample(selected_doc["title"][9:], result, 1)
                idoc += 1
        return result

    #
    # Build a sample of wikipedia api links for documents regarding
    # the content of the _topic_list instance variable
    # result is a python dictionary where "topic" feature is a list
    # of relevant url to get document in json
    #
    def get_sample(self):
        result = {}
        for topic in self._topic_list:
            res = self.get_doc_sample(topic, None, self._nb_doc_per_topic)
            result[topic] = res
        return result

    #
    # print the content of a sample for a given category
    #
    @staticmethod
    def print_topic(topic, structure):
        print("Results for category: " + str(topic) + " ==>")
        result = structure[topic]
        for l in result:
            print (l)

    #
    # print the content of a sample
    #
    def print_sample(self, structure):
        for topic in self._topic_list:
            self.print_topic(topic, structure)

    #
    # Download a document from the related media-wiki url
    # and save it in a unique name file
    #
    @staticmethod
    def download_document(lnk, target_dir):
        filename = uuid.uuid4()
        req = requests.get(lnk)
        if req.status_code == 200:
            stream = req.json()
            data = stream["query"]["pages"]
            for data_page in data.keys():
                data = data[data_page]["revisions"][0]["*"]
                break;
            f = open(target_dir + "/" + str(filename), "w")
            f.write(data.encode("utf-8"))
            f.close()

    #
    # Build the complete sample directory
    # structure is the following :
    #     ~/.wikisamples/dir_unique_name
    #           category1
    #               unique_name_file1_4_category_1
    #               unique_name_file2_4_category_1
    #               ...
    #           category2
    #               unique_name_file1_4_category_2
    #               unique_name_file2_4_category_2
    #               ...
    #
    def build_sample_tree(self, sample):
        result = uuid.uuid4()
        try:
            os.stat(self._target_dir + "/.wikisamples")
        except:
            os.mkdir(self._target_dir + "/.wikisamples")
        sample_dir = self._target_dir + "/.wikisamples/" + str(result)
        os.mkdir(sample_dir)
        # create a directory per category
        for topic in self._topic_list:
            os.mkdir(sample_dir + "/" + topic)
            doc_link_list = sample[topic]
            for lnk in doc_link_list:
                self.download_document(lnk, sample_dir + "/" + topic)
        return sample_dir

    #
    # Compress a given directory and save it in a tar.gz file
    #
    @staticmethod
    def make_tarfile(output_filename, source_dir):
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

    #
    # Save a complete sample (including train + test samples) in
    # a binary file as a sklearn.datasets.base.bunch compliant format
    #
    def make_cache_sample(self, train_path, test_path, cache_path):
        # Store a zipped pickle
        cache = dict(train=base.load_files(train_path, encoding='utf-8'),
                     test=base.load_files(test_path, encoding='utf-8'))

        compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
        with open(cache_path, 'wb') as f:
            f.write(compressed_content)



# Create a WikiSampleMaker instance
wsm = WikiSampleMaker()
print("Wiki train set generation in progres...")
# generate 15 document's url for each category
wsm._nb_doc_per_topic = 30
# build the train sample
train_sample = None
train_sample = wsm.get_sample()
wsm.print_sample(train_sample)
train_sample_dir = wsm.build_sample_tree(train_sample)
print("Wiki train set completed")

print("Wiki test set generation in progres...")
# generate between 5 and 20 document's url for each category
wsm._nb_doc_per_topic = None
# build the test sample
test_sample = None
test_sample = wsm.get_sample()
wsm.print_sample(test_sample)
test_sample_dir = wsm.build_sample_tree(test_sample)
print("Wiki test set completed")

print("Wiki binary sample generation in progress...")
# build the binary cache for document's sample
cache_dir = os.environ['HOME']+"/scikit_learn_data/wiki_sample.pkz"
wsm.make_cache_sample(train_sample_dir, test_sample_dir, cache_dir)
print("Wiki binary sample completed.")

# remove train and test directories
#shutil.rmtree(train_sample_dir)
#shutil.rmtree(test_sample_dir)

#wsm.make_tarfile(sample_dir + ".tar.gz", sample_dir)
