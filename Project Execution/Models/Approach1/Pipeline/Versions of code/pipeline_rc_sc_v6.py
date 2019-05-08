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
from sklearn.model_selection import train_test_split

# word2vec model
path1 = 'F:/somebody gonna get hurt/Github/Model/GoogleNews-vectors-negative300-normed.bin'
model = KeyedVectors.load(path1, mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
model.most_similar('stuff')
type(model['whistleblower'])

data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19_v2.csv')

train, test= train_test_split(data1, test_size=0.1, random_state=42)
data1=train
l =[]
for index, row in data1.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')
data1['res_vector']=l
#deleting the rows where difference vector could not be generated (because the word was not in vocabulary)
data1 = data1[data1.res_vector != 'not in dictionary']
data1=data1.reset_index(drop=True)
data1[data1.y ==1].count() #321
data1[data1.y ==0].count() #3961

#Preparing data
data11=data1
y1=data11.y
data1.describe()
y1.describe()
data11 = data11.drop("x1", axis=1)
data21=data11.drop("x2", axis=1)
data21=data21.drop("y", axis=1)
data21=data21.reset_index(drop=True)
xxx=pd.DataFrame(data21['res_vector'].values.tolist())
data31=xxx
data41=data31
data41.describe()
X1=data31


ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(X1, y1)
x1=list(x_resampled)
y1=list(y_resampled)
from collections import Counter
print(sorted(Counter(y_resampled).items()))

#Building SVM model and performing cross-validation
clf1 = svm.SVC(kernel='rbf', gamma=0.3) 
clf1.fit(x1,y1)

#test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files for try17/test17.csv')
#test1=pd.concat([test1,test])

test1=test
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


y_pred=clf1.predict(X_test)
test1['ypred']=y_pred

accuracy= accuracy_score(test1['y'],test1['ypred']) #0.9642857142857143
f1=f1_score(test1['y'],test1['ypred']) #0.7901234567901233
precision= precision_score(test1['y'],test1['ypred']) #0.7619047619047619
recall=recall_score(test1['y'],test1['ypred'])#0.8205128205128205

confusion_matrix(test1['y'],test1['ypred'])
#array([[427,  10],
#       [  7,  32]], dtype=int64)



test2=test1
test2=test2.drop("x1",axis=1)
test2=test2.drop("x2",axis=1)
test2=test2.drop("y",axis=1)
test2_new = test2[test2.ypred != 0]

#Data_version11 was created combining the above 2 steps
data2 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27_v2.csv')

#generating difference vector
l =[]
for index, row in data2.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')
data2['res_vector']=l
#deleting the rows where difference vector could not be generated (because the word was not in vocabulary)
data2 = data2[data2.res_vector != 'not in dictionary']
data2=data2.reset_index(drop=True)
data2[data2.y ==1].count() #152
data2[data2.y ==0].count() #114
data2[data2.y ==-1].count() #201

#Preparing data for visualization
data12=data2
y2=data12.y
data12.describe()
y2.describe()
data12 = data12.drop("x1", axis=1)
data22=data12.drop("x2", axis=1)
data22=data22.drop("y", axis=1)
data22=data22.reset_index(drop=True)
xxx=pd.DataFrame(data22['res_vector'].values.tolist())
data32=xxx
data42=data32
data42.describe()
X2=data32

ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(X2, y2)
#New predictors(x1) and predicted(y1)
x2=list(x_resampled)
y2=list(y_resampled)
from collections import Counter
print(sorted(Counter(y_resampled).items()))
clf2 = svm.SVC(kernel='rbf', gamma=0.75) 
clf2.fit(x2,y2)
#Predicting on another test set


xxx=pd.DataFrame(test2_new['res_vector'].values.tolist())
test13=xxx
test14=test13
test14.describe()
X_test=test13


y_pred=clf2.predict(X_test)
test2_new['ypred2']=y_pred
test2_new=test2_new.drop("ypred", axis=1)

result = pd.concat([test1, test2_new],  join='outer', axis=1)

result.to_csv('F:/somebody gonna get hurt/Github/Model/Results/Results_pipeline_rc_sc_v6.csv')
result_dataframe = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Results/SC/Results_pipeline_rc_sc_v6_formatted.csv')

confusion_matrix(result_dataframe['y'],result_dataframe['ypred_mcreated'])
#array([[448,   0],
#       [  6,  66]], dtype=int64)

accuracy= accuracy_score(result_dataframe['y'],result_dataframe['ypred_mcreated']) #0.9884615384615385
f1=f1_score(result_dataframe['y'],result_dataframe['ypred_mcreated']) #0.9565217391304348
precision= precision_score(result_dataframe['y'],result_dataframe['ypred_mcreated']) #1.0
recall=recall_score(result_dataframe['y'],result_dataframe['ypred_mcreated'])#0.9166666666666666