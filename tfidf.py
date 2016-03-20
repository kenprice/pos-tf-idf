from __future__ import division, unicode_literals
import math
from textblob import TextBlob as tb

import os
from os import listdir
from os.path import isfile, join

import codecs

import nltk
from nltk.tag.stanford import StanfordPOSTagger

DOC_DIR = 'txt'
POSTAGGER_JAR_PATH = r'/usr/local/bin/stanford-postagger-2015-04-20/stanford-postagger.jar'

# Fast model with 'good enough' ~97% accuracy
# Source: http://nlp.stanford.edu/software/pos-tagger-faq.shtml#train
MODEL_PATH = r'/usr/local/bin/stanford-postagger-2015-04-20/models/english-left3words-distsim.tagger'

# Java Heapspace Workaround
nltk.internals.config_java(options='-XX:+UseLargePages -Xmx6g')

################################################################################
# TF-IDF Computation
################################################################################

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


################################################################################
# COLLECT DOC TEXT
################################################################################

filepaths = [f for f in listdir(DOC_DIR) if isfile(join(DOC_DIR, f))]
filepaths.sort()
doclist = []

for filename in filepaths:
    f = codecs.open('./txt/' + filename, 'r', encoding='utf-8')
    doclist.append(f.read())


################################################################################
# EXTRACT NOUNS
################################################################################

st = StanfordPOSTagger(MODEL_PATH, POSTAGGER_JAR_PATH, java_options='-Xmx6g')

extracted_nouns_table = [[] for x in range(len(doclist))]

for i, doc in enumerate(doclist):
    print "==============================================="
    print filepaths[i]
    tagged_words = st.tag(doc.split())
    nouns = [t for t in tagged_words if t[1].startswith('NN')]
    print nouns
    print "==============================================="
