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

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

path = sys.argv[1]
file = open('2019EE10501.model', 'rb')
model = pickle.load(file)
vectorizer = pickle.load(file)
test = pd.read_csv(path)
Xtest = vectorizer.transform(test['profile']).toarray()
ytest = model.predict(Xtest)
df = pd.DataFrame(ytest,columns=['profession'])
df.to_csv(sys.argv[2],index=False)
#print(ytest)
