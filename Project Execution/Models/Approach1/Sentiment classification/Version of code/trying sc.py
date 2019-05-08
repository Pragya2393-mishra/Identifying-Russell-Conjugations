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


a = list(swn.senti_synsets('murderer'))
b= list(wn.synsets('murderer'))
len(a)
print(a)

sum_a=0
sum_a_l=[]
sum_a_n=0
sum_a_n_l=[]
sum_obj=0
sum_obj_l=[]
for i in a:
    sum_a = sum_a + i.pos_score()
    sum_a_l.append(i.pos_score())
    sum_a_n = sum_a_n + i.neg_score()
    sum_a_n_l.append(i.neg_score())
    sum_obj = sum_obj +i.obj_score()
    sum_obj_l.append(i.obj_score())
pos=sum_a/len(a)
neg=sum_a_n/len(a)
obj= sum_obj/len(a)
sum_a_l
#[0.0, 0.0, 0.25, 0.125, 0.0, 0.125]
sum_a_n_l
#[0.0, 0.0, 0.375, 0.0, 0.0, 0.0]


a = list(swn.senti_synsets('protest'))
b= list(wn.synsets('protest'))
len(a)
print(a)

sum_a=0
sum_a_l=[]
sum_a_n=0
sum_a_n_l=[]
sum_obj=0
sum_obj_l=[]
for i in a:
    sum_a = sum_a + i.pos_score()
    sum_a_l.append(i.pos_score())
    sum_a_n = sum_a_n + i.neg_score()
    sum_a_n_l.append(i.neg_score())
    sum_obj = sum_obj +i.obj_score()
    sum_obj_l.append(i.obj_score())
pos=sum_a/len(a)
neg=sum_a_n/len(a)
obj= sum_obj/len(a)





l=[]
average_score_word1=[]
average_score_word2=[]
for index, row in test1.iterrows():
    word1 = row['x1']
    word2 = row['x2']
    sum1_pos=0
    sum2_pos=0
    sum1_neg=0
    sum2_neg=0
    l_neg=[]
    word1_synset = list(swn.senti_synsets(word1))
    word2_synset = list(swn.senti_synsets(word2))
    for i in word1_synset:
        sum1_pos = sum1_pos + i.pos_score()
        sum1_neg = sum1_neg + i.neg_score()
    for i in word2_synset:
        sum2_pos = sum2_pos + i.pos_score()
        sum2_neg = sum2_neg + i.neg_score()
    try:
        average1= (sum1_pos - sum1_neg)/len(word1_synset)
        average2= (sum2_pos - sum2_neg)/len(word2_synset)
        l.append(average1 - average2)
        average_score_word1.append(average1)
        average_score_word2.append(average2)
    except:
        l.append('some error')
        average_score_word1.append('some error')
        average_score_word2.append('some error')

test1['l']=l
test1['average_score_word1'] = average_score_word1
test1['average_score_word2'] = average_score_word2


        
test1.to_csv('F:/somebody gonna get hurt/Github/Model/Results/tryingsc_using_sentiwordnet.csv')
    
