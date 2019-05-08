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
data = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav17.csv')

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

data[data.y ==1].count() #370
data[data.y ==0].count() #4266
#data.to_csv('F:/somebody gonna get hurt/Github/Model/data/datav14.csv')
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
classifier = svm.SVC(kernel='rbf', gamma=1)                             


def model_cv_predict(classifier, feature_vector_train, labels):
    estimator = Pipeline([("oversampler", oversamp),("classifier", classifier)])
    predictions = cross_val_predict(estimator, feature_vector_train, labels, cv=10)
    f1 = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='f1_macro')
    accuracy = cross_val_score(estimator, feature_vector_train, labels, cv=10)
    precision = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='precision_macro')
    recall = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='recall_macro')
    return f1,accuracy,precision,recall

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
np.savetxt("F:/somebody gonna get hurt/Github/Model/Results/Try22_svmr_v3/f1.csv", f1, delimiter=",")
np.savetxt("F:/somebody gonna get hurt/Github/Model/Results/Try22_svmr_v3/accuracy.csv", accuracy, delimiter=",")
np.savetxt("F:/somebody gonna get hurt/Github/Model/Results/Try22_svmr_v3/precision.csv", precision, delimiter=",")
np.savetxt("F:/somebody gonna get hurt/Github/Model/Results/Try22_svmr_v3/recall.csv", recall, delimiter=",")

f1=np.mean(f1) #0.7124735664437944
accuracy = np.mean(accuracy) #0.9432686936769198
precision = np.mean(precision) #0.9561274334089921
recall= np.mean(recall) #0.6495313410517751


oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=10)                             

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
f1=np.mean(f1) #0.4792181280370257
accuracy = np.mean(accuracy) #0.9201897296492143
precision = np.mean(precision) #0.46009486482460715
recall= np.mean(recall) #0.5

oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=0.5)                             

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
f1=np.mean(f1) #0.8129340692282401
accuracy = np.mean(accuracy) #0.9579355030907871
precision = np.mean(precision) #0.9588567817871093
recall= np.mean(recall) #0.7475942347334635

oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=0.05)                             

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
f1=np.mean(f1) #0.6077360055144765
accuracy = np.mean(accuracy) #0.8528836486184552
precision = np.mean(precision) #0.5920532795828823
recall= np.mean(recall) #0.6448490752519845

oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=0.3)                             

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
f1=np.mean(f1) #0.8416549429828895
accuracy = np.mean(accuracy) #0.9616020890742533
precision = np.mean(precision) #0.9352273821196047
recall= np.mean(recall) #0.7878445833767931

oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=0.3, C=1)                             

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
f1=np.mean(f1) #0.8416549429828895
accuracy = np.mean(accuracy) #0.9616020890742533
precision = np.mean(precision) #0.9352273821196047
recall= np.mean(recall) #0.7878445833767931

oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=0.3, C=5)                             

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
f1=np.mean(f1) #0.8428336787093637
accuracy = np.mean(accuracy) #0.9633266924852908
precision = np.mean(precision) #0.9664811153262243
recall= np.mean(recall) #0.7789095078520154

oversamp = RandomOverSampler(random_state=12)   
classifier = svm.SVC(kernel='rbf', gamma=0.3, C=12)                             

f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
f1=np.mean(f1) #0.8475386900919671
accuracy = np.mean(accuracy) #0.9633266924852908
precision = np.mean(precision) #0.9644042786921873
recall= np.mean(recall) #0.7831977539435402