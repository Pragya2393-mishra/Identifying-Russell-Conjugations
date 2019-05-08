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
data = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27_v3.csv')

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
data[data.y ==1].count() #152
data[data.y ==0].count() #114
data[data.y ==-1].count() #201

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
X=data3

clf = svm.SVC(kernel='rbf', gamma=0.75)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.6923459416400594
recall= np.mean(recall) #0.6583295359611149
accuracy=np.mean(accuracy) #0.6797324306898774
f1=np.mean(f1) #0.6628085913339266

clf = svm.SVC(kernel='rbf', gamma=0.75) 
clf.fit(X,y)
#Predicting on another test set
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files/test18.csv')
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
X_test=test13


y_pred=clf.predict(X_test)
test1['y-pred']=y_pred
test1= test1.drop("res_vector", axis=1)
test1 = test1.replace(np.nan, 0, regex=True)
test1 = test1.replace(-1, 1, regex=True)

confusion_matrix(test1['y'],test1['y-pred'])
#array([[10,  1],
#       [ 5, 35]], dtype=int64)
accuracy= accuracy_score(test1['y'],test1['y-pred']) #0.8823529411764706
f1=f1_score(test1['y'],test1['y-pred']) #0.9210526315789473
precision= precision_score(test1['y'],test1['y-pred']) #0.9722222222222222
recall=recall_score(test1['y'],test1['y-pred'])#0.875

test1.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Sentiment classification/Final version/SC_Test18_data27v3.csv')

