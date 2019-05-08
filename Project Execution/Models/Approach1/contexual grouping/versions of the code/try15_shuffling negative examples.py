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


#not working

#data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/negative_samples_final.csv')
#def shuffle(df, n=1, axis=0):     
#    df = df.copy()
#    for _ in range(n):
#        df.apply(np.random.shuffle, axis=axis)
#    return df
#
#data.loc[100]
#new_d.loc[100]

new_d = shuffle(data)
new_d.to_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/negative_samples_final_shuffled.csv')

