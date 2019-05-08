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
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.datasets.samples_generator import make_blobs
#from pandas.tools.plotting import parallel_coordinates
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

#creating negative samples
d= pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/words_alpha.txt')
d.describe()
d1=d
type(d)

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df
new_d = shuffle(d)
d.loc[10000]
new_d.loc[10000]
data= pd.concat([d, new_d], axis=1)
data.to_csv('F:/somebody gonna get hurt/Github/Model/negative samples1.csv')

#creating switched positive examples
data = pd.read_csv('F:/somebody gonna get hurt/Github/Literature survey/Collecting Russell Conjugations/Data/Data_version10.csv')
new_data1= pd.DataFrame(data['x2'])
new_data2= pd.DataFrame(data['x1'])
pos_samples = pd.concat([new_data1, new_data2], axis=1)
pos_samples.to_csv('F:/somebody gonna get hurt/Github/Model/positive_samples_final.csv')

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
data1 = data1.drop("x1", axis=1)
data2=data1.drop("x2", axis=1)
data2=data2.reset_index(drop=True)
xxx=pd.DataFrame(data2['res_vector'].values.tolist())
data2=data2.join(xxx)
cn=['y','res_vector']
for i in range(300):
    cn.append("column"+ str(i+1))
data2.columns=cn
data3=data2
data3=data3.drop("res_vector", axis=1)
data4=data3

#Reduction of data into two dimensions
normalized_X = preprocessing.normalize(data3)
pca = PCA(n_components=2)
data3[[0,1]]=pd.DataFrame(pca.fit_transform(normalized_X))
#visualization of data in two dimensions
fig = plt.figure()
plt.scatter(data3[data3.y==0][0], data3[data3.y==0][1], label='nonRC', c='red')
plt.scatter(data3[data3.y==1][0], data3[data3.y==1][1], label='RC', c='blue')
plt.legend()
plt.show()
fig.savefig('F:/somebody gonna get hurt/Github/Model/try16vis1.png')
#Reduction of data in three dimension
normalized_X = preprocessing.normalize(data4)
pca = PCA(n_components=3)
data4[[0,1,2]]=pd.DataFrame(pca.fit_transform(normalized_X))
#Visualization of data in three dimension
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data4[data4.y==0][0],data4[data4.y==0][1],data4[data4.y==0][2], c='black', label ='nonRC')
ax.scatter(data4[data4.y==0][0],data4[data4.y==0][1],data4[data4.y==0][2], c='yellow', label ='RC')
plt.legend()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
plt.savefig('F:/somebody gonna get hurt/Github/Model/try16vis2.png')
#Model building

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

#Building logistic regression and performing cross-validation
clf = LogisticRegression(C=100)
f1= cross_val_score(clf,x1,y1,cv=10,scoring='f1_macro')
recall=cross_val_score(clf,x1,y1,cv=10,scoring='recall_macro')
precision= cross_val_score(clf,x1,y1,cv=10,scoring='precision_macro')
accuracy =cross_val_score(clf,x1,y1,cv=10)
precision= np.mean(precision) #0.5724001137757017
recall= np.mean(recall) #0.5718540240016272
accuracy=np.mean(accuracy) #0.5718540240016272
f1=np.mean(f1) #0.5710759095584822

#Predicting results on test dt
a=list(X_test['res_vector'])
clf.fit(x1,y1)
y_pred=clf.predict(a)
X_test['y-pred']=y_pred
X_test.to_csv('F:/somebody gonna get hurt/Github/Model/try16result.csv')


