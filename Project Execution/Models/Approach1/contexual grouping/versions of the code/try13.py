# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:19:34 2019

@author: pragy
"""

import os
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from os.path import abspath, join, dirname
from inspect import getsourcefile
import nltk
import requests 
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from nltk.parse.stanford import StanfordDependencyParser
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/Data_version9.csv')

from sklearn.model_selection import train_test_split
X=data
X_train, X_test= train_test_split( X, test_size=0.2, random_state=42)
X_train.describe()
#                y
#count  291.000000
#mean     0.484536
#std      0.500622
#min      0.000000
#25%      0.000000
#50%      0.000000
#75%      1.000000
#max      1.000000
X_train[X_train.y ==1].count()
#x1            141
#x2            141
#y             141
#dtype: int64

X_train[X_train.y ==0].count()
#x1            150
#x2            150
#y             150
#dtype: int64

X_test.describe()
#              y     
#count  72.00000  
#mean    0.62500   
#std     0.48752   
#min     0.00000   
#25%     0.00000   
#50%     1.00000   
#75%     1.00000   
#max     1.00000   
X_test[X_test.y ==1].count()
#x1            45
#x2            45
#y             45
#dtype: int64

X_test[X_test.y ==0].count()
#x1            27
#x2            27
#y             27
#dtype: int64

# Feed a word2vec with the ingredients
path1 = 'F:/somebody gonna get hurt/Github/Model/GoogleNews-vectors-negative300-normed.bin'
model = KeyedVectors.load(path1, mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
model.most_similar('stuff')
type(model['whistleblower'])


l =[]
for index, row in X_train.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')

l
X_train1=X_train
X_train['res_vector']=l
#X_train.to_csv('F:/somebody gonna get hurt/Github/Model/secondmodelresvectors1.csv')
X_train = X_train[X_train.res_vector != 'not in dictionary']


l =[]
for index, row in X_test.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')

l
X_test['res_vector']=l
#X_test.to_csv('F:/somebody gonna get hurt/Github/Model/secondmodelresvectors12.csv')
X_test = X_test[X_test.res_vector != 'not in dictionary']



X = list(X_train['res_vector'])
X_res = list(X_test['res_vector'])



target = X_train.y
type(target)
target.describe()


clf = LogisticRegression(C=100)

clf.fit(X, target)

y_pred = clf.predict(X_res)
accuracy_score(X_test['y'],y_pred) #0.6666666666666666
precision_score(X_test['y'],y_pred) #0.8
recall_score(X_test['y'],y_pred) #0.6222222222222222
f1_score(X_test['y'],y_pred) #0.7000000000000001

confusion_matrix(X_test['y'],y_pred)
#array([[20,  7],
#       [17, 28]], dtype=int64)

X_test['y-pred']=y_pred
X_test.to_csv('F:/somebody gonna get hurt/Github/Model/try13result.csv')

