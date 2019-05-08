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

#Data_version11 was created combining the above 2 steps
data = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27.csv')

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
data4=data3
data4.describe()
X=data3

oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=0.8)                               
def model_cv_predict(classifier, feature_vector_train, labels):
    estimator = Pipeline([("oversampler", oversamp),("classifier", classifier)])
    predictions = cross_val_predict(estimator, feature_vector_train, labels, cv=10)
    f1 = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='f1_macro')
    accuracy = cross_val_score(estimator, feature_vector_train, labels, cv=10)
    precision = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='precision_macro')
    recall = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='recall_macro')
    return f1,accuracy,precision,recall

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)     
                                       
clf = svm.SVC(kernel='rbf', gamma=0.3)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.6494906403871074
recall= np.mean(recall) #0.6203544494720965
accuracy=np.mean(accuracy) #0.633262443438914
f1=np.mean(f1) #0.6180885735441771

clf = GaussianNB()  
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.6552362249988668
recall= np.mean(recall) #0.6425921137685843
accuracy=np.mean(accuracy) #0.646948717948718
f1=np.mean(f1) #0.6403394067264574

clf = svm.SVC(kernel='rbf', gamma=0.3)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.4188408952585522
recall= np.mean(recall) #0.5356115779645192
accuracy=np.mean(accuracy) #0.5972986425339367
f1=np.mean(f1) #0.4597962891580122

clf = svm.SVC(kernel='rbf', gamma=0.5)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.6882597973547967
recall= np.mean(recall) #0.6592286145227322
accuracy=np.mean(accuracy) #0.667027149321267
f1=np.mean(f1) #0.6587243160605871

clf = svm.SVC(kernel='rbf', gamma=0.8)                              
f1= cross_val_score(clf,X,y,cv=20,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=20,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=20,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=20)
precision= np.mean(precision) #0.6917711591535682
recall= np.mean(recall) #0.6689966242907419
accuracy=np.mean(accuracy) #0.6827541478129714
f1=np.mean(f1) #0.6695652649027997

clf = svm.SVC(kernel='rbf', gamma=1)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.6830868490150246
recall= np.mean(recall) #0.6602959132370898
accuracy=np.mean(accuracy) #0.6767933634992458
f1=np.mean(f1) #0.6615971570385214

clf = svm.LinearSVC(multi_class ='ovr', penalty= 'l2', loss='squared_hinge', dual=True)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.42591364414640276
recall= np.mean(recall) #0.4757376283846872
accuracy=np.mean(accuracy) #0.5268099547511313
f1=np.mean(f1) #0.4352087722276964

clf = gc(learning_rate=0.05)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.5473342079989677
recall= np.mean(recall) #0.5404316598434246
accuracy=np.mean(accuracy) #0.5697692307692308
f1=np.mean(f1) #0.5411940633064972

clf = rf()                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.4879462242455018
recall= np.mean(recall) #0.44232097967392087
accuracy=np.mean(accuracy) #0.5182699849170438
f1=np.mean(f1) #0.46658301417285786

clf = rf(n_estimators =75)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.5517254134378881
recall= np.mean(recall) #0.5367553688141923
accuracy=np.mean(accuracy) #0.5874162895927602
f1=np.mean(f1) #0.5355404565693715

clf = rf(n_estimators =1000)                              
f1= cross_val_score(clf,X,y,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,X,y,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,X,y,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,X,y,cv=10)
precision= np.mean(precision) #0.5517254134378881
recall= np.mean(recall) #0.5367553688141923
accuracy=np.mean(accuracy) #0.5874162895927602
f1=np.mean(f1) #0.5355404565693715


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
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files for try17/test11.csv')
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

confusion_matrix(test1['y'],y_pred, labels=[-1, 0, 1])
#array([[7, 1, 0],
#       [3, 9, 3],
#       [0, 2, 7]], dtype=int64)
accuracy= accuracy_score(test1['y'],y_pred) #0.71875

test1['y-pred']=y_pred
test1.to_csv('F:/somebody gonna get hurt/Github/Model/Results/Results_Test11_data27_sc_v2.csv')

