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

data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/Data_version8.csv')
test_data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/test_random2.csv')

from sklearn.model_selection import train_test_split
X=data
X_train, X_test= train_test_split( X, test_size=0.1, random_state=42)
X_train[X_train.y ==1].count()



# Target variable 


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

l =[]
for index, row in test_data.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')

l
test_data['res_vector']=l
#X_test.to_csv('F:/somebody gonna get hurt/Github/Model/secondmodelresvectors12.csv')
test_data = test_data[test_data.res_vector != 'not in dictionary']

X = list(X_train['res_vector'])
#X_res = list(X_test['res_vector'])
#X_res = list(test_data['res_vector'])



frames =[test_data, X_test]
test=pd.concat(frames)
X_res = list(test['res_vector'])
test.describe()


target = X_train.y
type(target)
target.describe()

#result = sm.Logit(target, X).fit().summary()
#coeff = sm.Logit(target, X).fit().params
#pval= sm.Logit(target, X).fit().pvalues

#a= pd.concat([coeff, pval], axis=1)
#type(a)
#a.to_csv('F:/somebody gonna get hurt/Github/Model/Firstmodelresult.csv')

clf = LogisticRegression(C=100)

clf.fit(X, target)

y_pred = clf.predict(X_res)
accuracy_score(test['y'],y_pred) #0.7692307692307693
precision_score(test['y'],y_pred) #0.7307692307692307
recall_score(test['y'],y_pred) #0.7916666666666666
f1_score(test['y'],y_pred) #0.76

confusion_matrix(test['y'],y_pred)
#array([[21,  7],
#       [ 5, 19]], dtype=int64)

test['y-pred']=y_pred
test.to_csv('F:/somebody gonna get hurt/Github/Model/try12result.csv')

type(X_test['y'])
q=pd.Series(X_test['y'])
type(q)

w=pd.Series(y_test)
type(w)

e=pd.concat([q,w], axis=1)