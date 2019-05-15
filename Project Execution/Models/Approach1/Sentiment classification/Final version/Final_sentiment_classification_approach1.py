# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:19:34 2019

@author: pragy
"""

import numpy as np
import pandas as pd
from sklearn import svm
from gensim.models import KeyedVectors
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
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
    
    data=import_data() #import data_sentiment_classification.csv
    test=import_data()
    ###################################################################################################################################
    
    # word2vec model
    path1 = 'F:/somebody gonna get hurt/Github/Project Execution/GoogleNews-vectors-negative300-normed.bin'
    model = KeyedVectors.load(path1, mmap='r')
    model.syn0norm = model.syn0  # prevent recalc of normed vectors
    model.most_similar('stuff')
    
    ###################################################################################################################################
    
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
    
    data= diff_vector(data)
    test =diff_vector(test)
    ##################################################################################################################################
    
    #Modeling
    
    #Preparing data for visualization
    data_original=data
    y=data.y #training predicted label
    X=pd.DataFrame(data['res_vector'].values.tolist()) #training predictor variables
    
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
    classifier = svm.SVC(kernel='rbf', gamma=0.75) 
    f1,accuracy,precision, recall= model_cv_predict(classifier, X,y)
    
    f1_score_cv=np.mean(f1) #0.6011826518816704
    accuracy_score_cv = np.mean(accuracy) #0.7063482652613088
    precision_score_cv = np.mean(precision) #0.6086768858388877
    recall_score_cv= np.mean(recall)#0.6007130124777185
    
    #Building Classifier
    ros = RandomOverSampler(random_state=12)
    x_resampled, y_resampled = ros.fit_resample(X, y)
    x1=list(x_resampled)
    y1=list(y_resampled)
    print(sorted(Counter(y_resampled).items()))
    clf = svm.SVC(kernel='rbf', gamma=0.75) 
    clf.fit(x1,y1)
    
    #Predicting on Test set
    
    test_original=test
    X_test=pd.DataFrame(test['res_vector'].values.tolist())
    
    
    y_pred=clf.predict(X_test)
    test['y-pred']=y_pred
    test= test.drop("res_vector", axis=1)
        
    confusion_matrix(test['y'],test['y-pred'])
    #array([[ 9, 20],
    #       [ 4, 11]], dtype=int64)
    accuracy= accuracy_score(test['y'],test['y-pred']) #0.45454545454545453
    f1=f1_score(test['y'],test['y-pred']) #0.47826086956521735
    precision= precision_score(test['y'],test['y-pred']) #0.3548387096774194
    recall=recall_score(test['y'],test['y-pred'])#0.7333333333333333
    
    
if __name__ == "__main__":
    main()
