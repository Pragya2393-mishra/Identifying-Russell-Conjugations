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
pca = PCA(n_components=5)
hello=pd.DataFrame(pca.fit_transform(normalized_X))
hello.describe()
hello['y']=y

x_mean = np.mean(hello[0])
y_mean = np.mean(hello[1])
z_mean= np.mean(hello[2])
z2_mean = np.mean(hello[3])
z3_mean = np.mean(hello[4])


from math import sqrt
import math
def euclidean_dist(x,y,z,z2,z3,xmean,ymean,zmean, z2mean, z3mean):
    a= math.pow(x-xmean,2)
    print(a)
    b=math.pow(y-ymean,2)
    print(b)
    c= math.pow(z-zmean,2)
    d= math.pow(z2-z2mean,2)
    e= math.pow(z3-z3mean,2)
    dist = sqrt(a +b +c+d+e)
    return dist



l=[]
for i in range(len(hello)):
    l.append(euclidean_dist(hello[0][i],hello[1][i],hello[2][i],hello[3][i],hello[4][i], x_mean,y_mean, z_mean,z2_mean, z3_mean))

hello['dist']=l
RC_dist = hello[y==1]['dist']
nonRC_dist = hello[y==0]['dist']

RC_dist.to_csv('F:/somebody gonna get hurt/Github/Model/Distances/RC_dist_PCA5.csv')
nonRC_dist.to_csv('F:/somebody gonna get hurt/Github/Model/Distances/nonRC_dist_PCA5.csv')

#bins = np.linspace(-10, 10, 100)
fig=plt.figure
n1, bins1, patches1 = plt.hist(RC_dist, alpha=0.5, bins=18,  label='RC', color='black')
n2, bin2, patches2 = plt.hist(nonRC_dist, alpha=0.5,bins=bins1, label='Non-RC', color='blue')
plt.legend(loc='upper right')
plt.show()



    

fig=plt.figure()
print('creating subplot')
ax=fig.add_subplot(111)
print('creating boxplot')
bp=ax.hist(nonRC_dist, bins=30, color='brown')
plt.title('Non RC_dist')
plt.xlabel('nonRC_distance')

plt.ylabel('Frequency')
plt.show()
fig.savefig('F:/somebody gonna get hurt/Github/Model/nonRC_dist.png')

#visualization of data in two dimensions
fig = plt.figure()
plt.scatter(hello[y==0][0], hello[y==0][1], label='nonRC', c='red')
plt.scatter(hello[y==1][0], hello[y==1][1], label='RC', c='blue')
plt.legend()
plt.show()
fig.savefig('F:/somebody gonna get hurt/Github/Model/try19vis1.png')


#Reduction of data in three dimension
normalized_X = preprocessing.normalize(X)
pca = PCA(n_components=3)
hello1=pd.DataFrame(pca.fit_transform(normalized_X))
hello1.describe()
hello1['y']=y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hello1[y==0][0],hello1[y==0][1],hello1[y==0][2], c='black', label ='nonRC')
ax.scatter(hello1[y==1][0],hello1[y==1][1],hello1[y==1][2], c='yellow', label ='RC')
plt.legend()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
plt.savefig('F:/somebody gonna get hurt/Github/Model/try19vis2.png')

