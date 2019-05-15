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
import tkinter as tk
from tkinter.filedialog import askopenfilename
from collections import Counter

def main():

    ###################################################################################################################################
    #Reading Data
    def import_data():
        root = tk.Tk()
        name = askopenfilename(title = "Select file",filetypes = (("csv",["*.csv",'*.txt']),("all files","*.*")))
        root.withdraw()        
        df= pd.read_csv(name, sep=',')
        return df
    
    data1=import_data() #import data_contexual_denotational_grouping.csv
    test1=import_data() #import Validation set.csv
    data2= import_data() #import data_sentiment_classifications.csv
    
    ###################################################################################################################################
    
    # word2vec model
    path1 = 'F:/somebody gonna get hurt/Github/Project Execution/GoogleNews-vectors-negative300-normed.bin'
    model = KeyedVectors.load(path1, mmap='r')
    model.syn0norm = model.syn0  # prevent recalc of normed vectors
    model.most_similar('stuff')
    
    #####################################################################################################################################
    
    #Data prep
    #creating difference vectors
    def diff_vector(data):
        l =[]
        for index, row in data.iterrows():
            #print(row['x1'], row['x2']) 
            try:
                l.append(model[row['x1']]-model[row['x2']])
            except:
                l.append('not in dictionary')
        data['res_vector']=l
        data = data[data.res_vector != 'not in dictionary']
        data=data.reset_index(drop=True)
        return data    
    
    data1= diff_vector(data1)
    data2= diff_vector(data2)
    test1= diff_vector(test1)
    
    #####################################################################################################################
    
    #Modeling Classifier 1 - contexual and denotational grouping
    
    #Preparing data1 for modeling
    data1_original=data1
    y1=data1.y #training predicted label
    X1=pd.DataFrame(data1['res_vector'].values.tolist()) #training predictor variables

    #Creating a pipeline of oversampling and cross-validation                           
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
    
    #Building Classifier 1
    ros = RandomOverSampler(random_state=12)
    x_resampled, y_resampled = ros.fit_resample(X1, y1)
    x1=list(x_resampled)
    y1=list(y_resampled)
    print(sorted(Counter(y_resampled).items()))
    clf1 = svm.SVC(kernel='rbf', gamma=0.3) 
    clf1.fit(x1,y1)
    
    #Testing classifier 1 on Validation set
    
    #Data prep of Validation set
    test1_original=test1
    X_test=pd.DataFrame(test1['res_vector'].values.tolist())#predictor variables of validation set
    
    #Using clf1 to predict on Validation data
    y_pred=clf1.predict(X_test)
    test1['ypred']=y_pred
    
    #Performance metric of classifier 1 on validation set
    accuracy= accuracy_score(test1['y'],test1['ypred']) # 0.7272727272727273
    f1=f1_score(test1['y'],test1['ypred']) #0.7000000000000001
    precision= precision_score(test1['y'],test1['ypred']) #0.56
    recall=recall_score(test1['y'],test1['ypred'])#0.9333333333333333
    confusion_matrix(test1['y'],test1['ypred'])
    #array([[18, 11],
    #       [ 1, 14]], dtype=int64)
       
    test2 = test1[test1.ypred != 0] #test set for classifier 2
    ####################################################################################################################################
        
    # Modeling Classifier 2
               
    #Preparing data
    data2_original=data2
    y2=data2.y
    X2=pd.DataFrame(data2['res_vector'].values.tolist())
    
    #Creating pipeline for oversampling and cross-validation
    def model_cv_predict(classifier, feature_vector_train, labels):
        estimator = Pipeline([("oversampler", oversamp),("classifier", classifier)])
        predictions = cross_val_predict(estimator, feature_vector_train, labels, cv=10)
        f1 = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='f1_macro')
        accuracy = cross_val_score(estimator, feature_vector_train, labels, cv=10)
        precision = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='precision_macro')
        recall = cross_val_score(estimator, feature_vector_train, labels, cv=10,scoring='recall_macro')
        return f1,accuracy,precision,recall

    oversamp = RandomOverSampler(random_state=12)   
    classifier = svm.SVC(kernel='rbf', gamma=0.75) 
    f1,accuracy,precision, recall= model_cv_predict(classifier, X2,y2)
    
    #cross-validation scores
    f1_score_cv=np.mean(f1) #0.6011826518816704
    accuracy_score_cv = np.mean(accuracy) #0.7063482652613088
    precision_score_cv = np.mean(precision) #0.6086768858388877
    recall_score_cv= np.mean(recall)#0.6007130124777185
    
    #Building classifier 2
    ros = RandomOverSampler(random_state=0)
    x_resampled, y_resampled = ros.fit_resample(X2, y2)
    x2=list(x_resampled)
    y2=list(y_resampled)
    print(sorted(Counter(y_resampled).items()))
    clf2 = svm.SVC(kernel='rbf', gamma=0.75) 
    clf2.fit(x2,y2)

    #Using clf2 on our test data
    
    X_test=pd.DataFrame(test2['res_vector'].values.tolist())
    y_pred=clf2.predict(X_test)
    test2['ypred2']=y_pred
    test2=test2.drop("ypred", axis=1)

    ##############################################################################################################################################
    
    #combining the results of classifier 2 with the original validation set
    test2=test2.drop("x1",axis=1)
    test2=test2.drop("x2",axis=1)
    test2=test2.drop("y",axis=1)
    test2=test2.drop("res_vector",axis=1)
    
    result = pd.concat([test1, test2],  join='outer', axis=1)
    result2= result.copy()
    result2 = result2.drop("res_vector",axis=1)
    result2 = result2.replace(np.nan, 0, regex=True)
    result2 = result2.drop("ypred",axis=1)
    
    #Final Validation results
    confusion_matrix(result2['y'],result2['ypred2'])
    #array([[25,  4],
    #       [ 4, 11]], dtype=int64)

    accuracy= accuracy_score(result2['y'],result2['ypred2']) #0.8181818181818182
    f1=f1_score(result2['y'],result2['ypred2']) #0.7333333333333333
    precision= precision_score(result2['y'],result2['ypred2']) #0.7333333333333333
    recall=recall_score(result2['y'],result2['ypred2'])#0.7333333333333333

    result2.to_csv('F:/somebody gonna get hurt/Github/Project Execution/Models/Approach1/Pipeline/Final version/pipeline_result.csv')

if __name__ == "__main__":
    main()
