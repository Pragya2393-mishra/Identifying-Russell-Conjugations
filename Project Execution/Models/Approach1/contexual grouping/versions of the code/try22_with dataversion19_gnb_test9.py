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

#Data_version11 was created combining the above 2 steps
data = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19.csv')

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
data[data.y ==1].count() #372
data[data.y ==0].count() #4398

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
classifier = GaussianNB()                               
def model_cv_predict(classifier, feature_vector_train, labels):
    estimator = Pipeline([("oversampler", oversamp),("classifier", classifier)])
    predictions = cross_val_predict(estimator, feature_vector_train, labels, cv=10)
    f1 = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='f1_macro')
    accuracy = cross_val_score(estimator, feature_vector_train, labels, cv=10)
    precision = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='precision_macro')
    recall = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='recall_macro')
    return f1,accuracy,precision,recall

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
np.savetxt("F:/somebody gonna get hurt/Github/Model/Results/Try21_gnb_cv/f1.csv", f1, delimiter=",")
np.savetxt("F:/somebody gonna get hurt/Github/Model/Results/Try21_gnb_cv/accuracy.csv", accuracy, delimiter=",")
np.savetxt("F:/somebody gonna get hurt/Github/Model/Results/Try21_gnb_cv/precision.csv", precision, delimiter=",")
np.savetxt("F:/somebody gonna get hurt/Github/Model/Results/Try21_gnb_cv/recall.csv", recall, delimiter=",")

f1=np.mean(f1) #0.685981864905463
accuracy = np.mean(accuracy) #0.8923653785560495
precision = np.mean(precision) #0.6646483726956962
recall= np.mean(recall) #0.7221304431373133




#Performing oversampling to offset the effect of negative bias
ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(X, y)
#New predictors(x1) and predicted(y1)
x1=list(x_resampled)
y1=list(y_resampled)
from collections import Counter
print(sorted(Counter(y_resampled).items()))

#Building SVM model and performing cross-validation
clf = GaussianNB()


#Predicting results on test df

clf.fit(x1,y1)

#Predicting on another test set
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files for try17/test9.csv')
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
accuracy= accuracy_score(test1['y'],y_pred) #0.75
f1=f1_score(test1['y'],y_pred) #0.75
precision= precision_score(test1['y'],y_pred) #0.6521739130434783
recall=recall_score(test1['y'],y_pred) #0.8823529411764706
test1['y-pred']=y_pred
test1.to_csv('F:/somebody gonna get hurt/Github/Model/Results/Results_Test9_data19.csv')