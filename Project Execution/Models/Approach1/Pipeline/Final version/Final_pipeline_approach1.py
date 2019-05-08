# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:19:34 2019

@author: pragy
"""
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split
import copy

# word2vec model
path1 = 'F:/somebody gonna get hurt/Github/Model/GoogleNews-vectors-negative300-normed.bin'
model = KeyedVectors.load(path1, mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
model.most_similar('stuff')
type(model['whistleblower'])

data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19_v3.csv')
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
data1[data1.y ==1].count() #351
data1[data1.y ==0].count() #4398

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
                              
def model_cv_predict(classifier, feature_vector_train, labels):
    estimator = Pipeline([("oversampler", oversamp),("classifier", classifier)])
    predictions = cross_val_predict(estimator, feature_vector_train, labels, cv=10)
    f1 = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='f1_macro')
    accuracy = cross_val_score(estimator, feature_vector_train, labels, cv=10)
    precision = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='precision_macro')
    recall = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='recall_macro')
    return f1,accuracy,precision,recall

oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=0.3) 
f1,accuracy,precision, recall= model_cv_predict(classifier, X1,y1)

f1_score_cv=np.mean(f1) #0.7914249436945597
accuracy_score_cv = np.mean(accuracy) #0.9407896621673318
precision_score_cv = np.mean(precision) #0.9260995220105606
recall_score_cv= np.mean(recall) #0.7411924155318232

#Building SVM model 
ros = RandomOverSampler(random_state=12)
x_resampled, y_resampled = ros.fit_resample(X1, y1)
x1=list(x_resampled)
y1=list(y_resampled)
from collections import Counter
print(sorted(Counter(y_resampled).items()))
clf1 = svm.SVC(kernel='rbf', gamma=0.3) 
clf1.fit(x1,y1)


test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files/test9_20.csv')
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

#Using clf1 to predict on test data
y_pred=clf1.predict(X_test)
test1['ypred']=y_pred

accuracy= accuracy_score(test1['y'],test1['ypred']) # 0.7272727272727273
f1=f1_score(test1['y'],test1['ypred']) #0.7000000000000001
precision= precision_score(test1['y'],test1['ypred']) #0.56
recall=recall_score(test1['y'],test1['ypred'])#0.9333333333333333
confusion_matrix(test1['y'],test1['ypred'])
#array([[18, 11],
#       [ 1, 14]], dtype=int64)

#Data prep for next model to predict on test data

test2=test1.copy()
test2=test2.drop('res_vector',axis=1)
test2.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Pipeline/Final versons/pipeline_data19v3_test9-20_data27v4_part1.csv')


test3=test1.copy()
test3=test3.drop("x1",axis=1)
test3=test3.drop("x2",axis=1)
test3=test3.drop("y",axis=1)
test2_new = test3[test3.ypred != 0]

data2 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27_v4.csv')

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
data2[data2.y ==1].count() 
data2[data2.y ==0].count() 

#Preparing data
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

#Using clf2 on our test data

xxx=pd.DataFrame(test2_new['res_vector'].values.tolist())
test13=xxx
X_test=test13


y_pred=clf2.predict(X_test)
test2_new['ypred2']=y_pred
test2_new=test2_new.drop("ypred", axis=1)

result = pd.concat([test1, test2_new],  join='outer', axis=1)
result1= result.drop("res_vector", axis=1)
result2= result1.copy()
result2 = result2.replace(np.nan, 0, regex=True)
result2 = result2.replace(-1, 1, regex=True)
result2 = result2.drop("ypred",axis=1)

confusion_matrix(result2['y'],result2['ypred2'])
#array([[25,  4],
#       [ 4, 11]], dtype=int64)

accuracy= accuracy_score(result2['y'],result2['ypred2']) #0.8181818181818182
f1=f1_score(result2['y'],result2['ypred2']) #0.7333333333333333
precision= precision_score(result2['y'],result2['ypred2']) #0.7333333333333333
recall=recall_score(result2['y'],result2['ypred2'])#0.7333333333333333

result2.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Pipeline/Final versons/pipeline_data19v3_test9-20_data27v4.csv')
