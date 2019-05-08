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
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.tools.plotting import parallel_coordinates
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error


#Data_version11 was created combining the above 2 steps
data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/Data_version13.csv')

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

#Reduction of data into two dimensions
normalized_X = preprocessing.normalize(X)
pca = PCA(n_components=2)
hello=pd.DataFrame(pca.fit_transform(normalized_X))
hello.describe()
hello['y']=y

x_mean = np.mean(hello[0])
sub1=x_mean
y_mean = np.mean(hello[1])
sub2=y_mean

from math import sqrt
import math
def euclidean_dist(x,y,xmean,ymean):
    a= math.pow(x-xmean,2)
    print(a)
    b=math.pow(y-ymean,2)
    print(b)
    dist = sqrt(a +b )
    return dist

l=[]
for i in range(len(hello)):
    l.append(euclidean_dist(hello[0][i],hello[1][i],x_mean,y_mean))

hello['dist']=l
X['dist']=l
data['dist']=l


#Splitting data into train and test
from sklearn.model_selection import train_test_split
X=data
X_train, X_test = train_test_split( X, test_size=0.4, random_state=42)
X_train[X_train.y ==1].count()
X_train[X_train.y ==0].count()
X_test[X_test.y ==1].count()
X_test[X_test.y ==0].count()


l=[]
for index, row in result.iterrows():
    print(row['res_vector'], row['dist']) 
    try:
        l.append(np.append(row['res_vector'], row['dist']))
    except:
        print('error')
        
h=l[1]

#Retrieving predictors(x) and predicted(y) variables
x1= X_train['res_vector']
x2=X_train['dist']
result = pd.concat([x1, x2], axis=1)
x=l
#x=result
x=list(x)
y= X_train['y']
y=list(y)


l=[]
for index, row in result.iterrows():
    print(row['res_vector'], row['dist']) 
    try:
        l.append(np.append(row['res_vector'], row['dist']))
    except:
        print('error')
        
h=l[1]


x1= X_test['res_vector']
x2=X_test['dist']
X_test['l']=l
result = pd.concat([x1, x2], axis=1)
x=l
y= X_train['y']

#Performing oversampling to offset the effect of negative bias
ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(x, y)
#New predictors(x1) and predicted(y1)
x1=list(x_resampled)
y1=list(y_resampled)
from collections import Counter
print(sorted(Counter(y_resampled).items()))

#Building logistic regression and testing on the test set
clf = GaussianNB()
a=list(X_test['l'])
clf.fit(x1,y1)
y_pred=clf.predict(a)
accuracy= accuracy_score(X_test['y'],y_pred) #0.908351409978308
f1=f1_score(X_test['y'],y_pred) #0.5677749360613811
precision= precision_score(X_test['y'],y_pred) #0.4723404255319149
recall=recall_score(X_test['y'],y_pred) # 0.7115384615384616
X_test['y-pred']=y_pred 
X_test.to_csv('F:/somebody gonna get hurt/Github/Model/try20result.csv')

#test1
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files for try17/test5.csv')
test_initial =test1
#test1=test1.drop("y",axis=1)
l =[]
for index, row in test1.iterrows():
    print(row['x1'], row['x2']) 
    try:
        l.append(model[row['x1']]-model[row['x2']])
    except:
        l.append('not in dictionary')
test1['res_vector']=l

#deleting the rows where difference vector could not be generated (because the word was not in vocabulary)
test1 = test1[test1.res_vector != 'not in dictionary']
test1=test1.reset_index(drop=True)

test11=test1
y=test11.y
test11.describe()
y.describe()
test11 = test11.drop("x1", axis=1)
test12=test11.drop("x2", axis=1)
test12=test12.drop("y", axis=1)
test12=test12.reset_index(drop=True)
xxx=pd.DataFrame(test12['res_vector'].values.tolist())
test13=xxx
test14=test13
test14.describe()
X=test13

#Reduction of test1 into two dimensions
normalized_X = preprocessing.normalize(X)
pca = PCA(n_components=2)
hello=pd.DataFrame(pca.fit_transform(normalized_X))
hello.describe()
hello['y']=y

x_mean = np.mean(hello[0])
y_mean = np.mean(hello[1])

from math import sqrt
import math
def euclidean_dist(x,y,xmean,ymean):
    a= math.pow(x-xmean,2)
    print(a)
    b=math.pow(y-ymean,2)
    print(b)
    dist = sqrt(a +b )
    return dist

l=[]
for i in range(len(hello)):
    l.append(euclidean_dist(hello[0][i],hello[1][i],sub1,sub2))

hello['dist']=l
X['dist']=l
test1['dist']=l

l=[]
for index, row in result.iterrows():
    print(row['res_vector'], row['dist']) 
    try:
        l.append(np.append(row['res_vector'], row['dist']))
    except:
        print('error')
        
h=l[1]


x1= test1['res_vector']
x2=test1['dist']
result = pd.concat([x1, x2], axis=1) 
test1['l']=l

y_pred=clf.predict(l)
test_initial['y-pred']=y_pred
test_initial.to_csv('F:/somebody gonna get hurt/Github/Model/test5result.csv')

accuracy_score(test1['y'],y_pred) #0.673469387755102
precision_score(test1['y'],y_pred) #0.6666666666666666
recall_score(test1['y'],y_pred) #0.2222222222222222
f1_score(test1['y'],y_pred) #0.3333333333333333

confusion_matrix(test1['y'],y_pred)
#array([[29,  2],
#       [14,  4]], dtype=int64)