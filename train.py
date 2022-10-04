import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import sklearn

import sys

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

import time

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets

import pickle
nltk.download('all')
train = pd.read_csv(sys.argv[1])
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),ngram_range=(1,1),stop_words='english',max_features=3000)
#vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words='english')
a = vectorizer.fit(train['profile'][:])
bow_vectors = a.transform(train['profile'][:])
X=bow_vectors.toarray()
y=train['profession'][:].values.tolist()
model = LogisticRegression(multi_class='multinomial',solver='saga',max_iter=50)#,class_weight='balanced')
model.fit(X[:],y[:])
pickle.dump(model,open(sys.argv[2],'wb'))
