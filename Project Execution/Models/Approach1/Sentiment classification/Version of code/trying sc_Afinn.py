# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:39:27 2019

@author: pragy
"""

import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import nltk
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files for try17/test11.csv')
Afinn_list = pd.read_csv('F:/somebody gonna get hurt/Github/Model/SentiWordNet/iFINN/AFINN/AFINN-111.txt', sep='\t', header=None)

l=[]
score1_l=[]
score2_l=[]
for index, row in test1.iterrows():
    word1 = row['x1']
    word2 = row['x2']
    score1=100
    score2=100
    for index1, row1 in Afinn_list.iterrows():
        if row1[0]==word1:
            score1=row1[1]
            
        if row1[0]==word2:
            score2=row1[1]
    if score1!=100:
        score1_l.append(score1)
    else:
        score1_l.append('not present')
    if score2!=100:
        score2_l.append(score2)
    else:
        score2_l.append('not present')
    if score1<100 and score2 <100:
        l.append(score1-score2)
    else:
        l.append('not present')
  



test1['l']=l
test1['score1']=score1_l
test1['score2']=score2_l



        
test1.to_csv('F:/somebody gonna get hurt/Github/Model/Results/tryingsc_using_AFinn.csv')
    
