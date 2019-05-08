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

data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/tobereversed.csv')
new_data1= pd.DataFrame(data['x2'])
new_data2= pd.DataFrame(data['x1'])
pos_samples = pd.concat([new_data1, new_data2], axis=1)


pos_samples.to_csv('F:/somebody gonna get hurt/Github/Model/synonyms_switched.csv')

