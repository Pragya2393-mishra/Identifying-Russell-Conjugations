# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:19:34 2019

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
from sklearn.ensemble import GradientBoostingClassifier as gc
from sklearn.linear_model import Perceptron as pc
from sklearn.ensemble import RandomForestClassifier as rf

# word2vec model
path1 = 'F:/somebody gonna get hurt/Github/Model/GoogleNews-vectors-negative300-normed.bin'
model = KeyedVectors.load(path1, mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
model.most_similar('stuff')
type(model['whistleblower'])

#Data_version11 was created combining the above 2 steps
data = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav22.csv')


#generating difference vector
l =[]
for index, row in data.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')
data['res_vector']=l
#deleting the rows where difference vector could not be generated (because the word was not in vocabulary)
data = data[data.res_vector != 'not in dictionary']
data=data.reset_index(drop=True)
data[data.y ==1].count() #371
data[data.y ==0].count() #133


#Preparing data for visualization
data1=data
y=data1.y
data1.describe()
y.describe()
data1 = data1.drop("x1", axis=1)
data2=data1.drop("x2", axis=1)
data2=data2.drop("y", axis=1)
data2=data2.reset_index(drop=True)
xxx=pd.DataFrame(data2['res_vector'].values.tolist())
data3=xxx
data4=data3
data4.describe()
X=data3


ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(X, y)
#New predictors(x1) and predicted(y1)
x1=list(x_resampled)
y1=list(y_resampled)
from collections import Counter
print(sorted(Counter(y_resampled).items()))

clf = svm.SVC(kernel='rbf', gamma=0.8)
clf.fit(X,y)
#Predicting on another test set
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files for try17/test12.csv')
l =[]
for index, row in test1.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')
test1['res_vector']=l

test1 = test1[test1.res_vector != 'not in dictionary']
test1=test1.reset_index(drop=True)
test11=test1
y=test11.y
test11.describe()
y.describe()
test11 = test11.drop("x1", axis=1)
test12=test11.drop("x2", axis=1)
test12=test12.drop("y", axis=1)
xxx=pd.DataFrame(test12['res_vector'].values.tolist())
test13=xxx
test14=test13
test14.describe()
X_test=test13

y_pred=clf.predict(X_test)
confusion_matrix(test1['y'],y_pred)
test1['y-pred']=y_pred

accuracy= accuracy_score(test1['y'],y_pred) #0.625
f1=f1_score(test1['y'],y_pred) #0.7391304347826086
precision= precision_score(test1['y'],y_pred) #0.5862068965517241
recall=recall_score(test1['y'],y_pred)#1.0
test1.to_csv('F:/somebody gonna get hurt/Github/Model/Results/Results_Test12_data22_sc_v1.csv')