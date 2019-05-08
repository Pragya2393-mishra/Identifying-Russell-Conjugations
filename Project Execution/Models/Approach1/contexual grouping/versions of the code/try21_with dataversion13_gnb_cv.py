# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:19:34 2019

@author: pragy
"""
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from inspect import getsourcefile
import nltk
import requests 
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

#Data_version11 was created combining the above 2 steps
data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/Data_version11.csv')

# word2vec model
path1 = 'F:/somebody gonna get hurt/Github/Model/GoogleNews-vectors-negative300-normed.bin'
model = KeyedVectors.load(path1, mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
model.most_similar('stuff')
type(model['whistleblower'])

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



#Splitting data into train and test
from sklearn.model_selection import train_test_split
X=data
X_train, X_test= train_test_split( X, test_size=0.2, random_state=42)
X_train[X_train.y ==1].count()
X_train[X_train.y ==0].count()
X_test[X_test.y ==1].count()
X_test[X_test.y ==0].count()

#Retrieving predictors(x) and predicted(y) variables
x= X_train['res_vector']
x=list(x)
y= X_train['y']
y=list(y)

#Performing oversampling to offset the effect of negative bias
ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(x, y)
#New predictors(x1) and predicted(y1)
x1=list(x_resampled)
y1=list(y_resampled)
from collections import Counter
print(sorted(Counter(y_resampled).items()))

#Building SVM model and performing cross-validation
clf = GaussianNB()
#f1= cross_val_score(clf,x1,y1,cv=10,scoring='f1_macro')
#recall=cross_val_score(clf,x1,y1,cv=10,scoring='recall_macro')
#precision= cross_val_score(clf,x1,y1,cv=10,scoring='precision_macro')
#accuracy =cross_val_score(clf,x1,y1,cv=10)
#precision= np.mean(precision) #0.5724001137757017
#recall= np.mean(recall) #0.5718540240016272
#accuracy=np.mean(accuracy) #0.5718540240016272
#f1=np.mean(f1) #0.5578434801390826

#Predicting results on test df
a=list(X_test['res_vector'])
clf.fit(x1,y1)
y_pred=clf.predict(a)
accuracy= accuracy_score(X_test['y'],y_pred) #0.9065520945220193
f1=f1_score(X_test['y'],y_pred) #0.6027397260273971
precision= precision_score(X_test['y'],y_pred) #0.5196850393700787
recall=recall_score(X_test['y'],y_pred) # 0.0.717391304347826
X_test['y-pred']=y_pred 
X_test.to_csv('F:/somebody gonna get hurt/Github/Model/try18result.csv')



#Predicting on another test set
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files for try17/test5.csv')
l =[]
for index, row in test1.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')
test1['res_vector']=l

test1 = test1[test1.res_vector != 'not in dictionary']
a=list(test1['res_vector'])
y_pred=clf.predict(a)
accuracy= accuracy_score(test1['y'],y_pred) #0.7755102040816326
f1=f1_score(test1['y'],y_pred) #0.6451612903225806
precision= precision_score(test1['y'],y_pred) #0.7692307692307693
recall=recall_score(test1['y'],y_pred) #0.5555555555555556
test1['y-pred']=y_pred
test1.to_csv('F:/somebody gonna get hurt/Github/Model/try18_dataversion13_test5_gnb_result.csv')

