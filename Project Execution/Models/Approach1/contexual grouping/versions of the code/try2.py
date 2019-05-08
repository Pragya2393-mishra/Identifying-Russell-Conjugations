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

data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/Data_version72.csv')

from sklearn.model_selection import train_test_split
X=data
X_train, X_test= train_test_split( X, test_size=0.2, random_state=42)
X_train[X_train.y ==1].count()



# Target variable 


# Feed a word2vec with the ingredients
path1 = 'F:/somebody gonna get hurt/Github/Model/GoogleNews-vectors-negative300-normed.bin'
model = KeyedVectors.load(path1, mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
model.most_similar('stuff')
type(model['whistleblower'])

#X_train['x'] = X_train[X_train.columns[:2]].apply(
#    lambda x: ','.join(x.dropna().astype(str).astype(str)),
#    axis=1
#)
#
#X_test['x'] = X_test[X_test.columns[:2]].apply(
#    lambda x: ','.join(x.dropna().astype(str).astype(str)),
#    axis=1
#)
#type(X_train['x'])
#type(X_test['x'])
#def resultant_vector(doc):
#    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
#    doc = [word for word in doc if word in model.wv.vocab]
#    print(doc)
#    x= np.mean(model[doc], axis=0)
#   
#    print('-----------------------------------------------------------------------------------------')
#    #print(x)
#    #print('-----------------------------------------------------------------------------------------')
#    return x

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
X_train.to_csv('F:/somebody gonna get hurt/Github/Model/secondmodelresvectors1.csv')
X_train = X_train[X_train.res_vector != 'not in dictionary']
y=target


l =[]
for index, row in X_test.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')

l
X_test['res_vector']=l
X_test.to_csv('F:/somebody gonna get hurt/Github/Model/secondmodelresvectors12.csv')
X_test = X_test[X_test.res_vector != 'not in dictionary']



X = list(X_train['res_vector'])
X_res = list(X_test['res_vector'])



target = X_train.y
type(target)
target.describe()

#result = sm.Logit(target, X).fit().summary()
#coeff = sm.Logit(target, X).fit().params
#pval= sm.Logit(target, X).fit().pvalues

a= pd.concat([coeff, pval], axis=1)
type(a)
a.to_csv('F:/somebody gonna get hurt/Github/Model/Firstmodelresult.csv')

clf = LogisticRegression(C=100)

clf.fit(X, target)

y_pred = clf.predict(X_res)
accuracy_score(X_test['y'],y_pred) #0.8918918918918919
precision_score(X_test['y'],y_pred) #0.925
recall_score(X_test['y'],y_pred) #0.8809523809523809
f1_score(X_test['y'],y_pred) #0.9024390243902439

confusion_matrix(X_test['y'],y_pred)

type(X_test['y'])
q=pd.Series(X_test['y'])
type(q)

w=pd.Series(y_test)
type(w)

e=pd.concat([q,w], axis=1)