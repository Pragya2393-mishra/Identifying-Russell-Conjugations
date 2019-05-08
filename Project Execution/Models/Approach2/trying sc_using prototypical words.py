# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:39:27 2019

@author: pragy
"""

import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import imblearn
import warnings
warnings.filterwarnings('ignore')
from random import shuffle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

# word2vec model
path1 = 'F:/somebody gonna get hurt/Github/Model/GoogleNews-vectors-negative300-normed.bin'
model = KeyedVectors.load(path1, mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
model.most_similar('stuff')
type(model['whistleblower'])

#good, evil, alpha=1
l= model['good'] - model['evil']
w2 =model['leader']-l
model.most_similar(positive=[w2])
#[('leader', 0.5994877815246582),
# ('evil', 0.5045323967933655),
# ('V_Hugo_Weaving', 0.4516620934009552),
# ('bloodthirsty_tyrant', 0.437578022480011),
# ('betrayer', 0.4211561679840088),
# ('evil_machinations', 0.4193650484085083),
# ('uber_villain', 0.4181157350540161),
# ('tyrannical_ruler', 0.41702014207839966),
# ('Afghan_warlord', 0.41607141494750977),
# ('Sheikh_Fadlallah', 0.41557633876800537)]

w2 =(model['leader'])-(2*l)
model.most_similar(positive=[w2])
#[('evil', 0.5948091745376587),
# ('V_Hugo_Weaving', 0.4546595811843872),
# ('evil_machinations', 0.4537809193134308),
# ('necromancer', 0.45304468274116516),
# ('demon_lord', 0.448440283536911),
# ('unspeakable_evil', 0.445115327835083),
# ('malevolent', 0.44001060724258423),
# ('tyrannical_ruler', 0.43630513548851013),
# ('evil_doer', 0.43616703152656555),
# ('bloodthirsty_tyrant', 0.4338264465332031)]

w2 =(model['leader'])-(3*l)
x=model.most_similar(positive=[w2])
#[('evil', 0.6159547567367554),
# ('necromancer', 0.4592245817184448),
# ('unspeakable_evil', 0.45902419090270996),
# ('malevolent', 0.45727407932281494),
# ('evil_machinations', 0.45589643716812134),
# ('demon_lord', 0.45242029428482056),
# ('Ahriman', 0.44416889548301697),
# ('V_Hugo_Weaving', 0.4439713954925537),
# ('evil_doer', 0.4412302076816559),
# ('tyrannical_ruler', 0.4327232241630554)]



result=pd.DataFrame()
alpha=[]
for i in np.arange(0.1,5,0.2):
    df=[]
    w2 =(model['leader'])-(i*l)
    x= model.most_similar(positive=[w2])
    df.append(x)
    result= result.append(df)

alpha=np.array(np.arange(0.1,5,0.2))
result['alpha']=alpha

#alpha 0.9 to 2.1 gives maximum number of potential russell conjugates. 
    
result.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Sentiment classification/Final version_pw/result_upto_5.csv')

#Lets try these alpha on another word like "obstinate"

result=pd.DataFrame()
alpha=[]
for i in np.arange(0.9,2.3,0.2):
    df=[]
    w2 =(model['obstinate'])-(i*l)
    x= model.most_similar(positive=[w2])
    df.append(x)
    result= result.append(df)

alpha=np.array(np.arange(0.9,2.3,0.2))
result['alpha']=alpha
#No russell conjugate identified
result.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Sentiment classification/Final version_pw/result_obstinate.csv')

result=pd.DataFrame()
alpha=[]
for i in np.arange(0.1,5,0.2):
    df=[]
    w2 =(model['obstinate'])-(i*l)
    x= model.most_similar(positive=[w2])
    df.append(x)
    result= result.append(df)

alpha=np.array(np.arange(0.1,5,0.2))
result['alpha']=alpha
#alpha 0.1 to 0.5 gives maximum number of russell conjugates.
result.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Sentiment classification/Final version_pw/result_obstinate_upto_5.csv')


#Lets try these alpha on another word like "disciplinarian"

result=pd.DataFrame()
alpha=[]
for i in np.arange(0.1,5,0.2):
    df=[]
    w2 =(model['disciplinarian'])-(i*l)
    x= model.most_similar(positive=[w2])
    df.append(x)
    result= result.append(df)

alpha=np.array(np.arange(0.1,5,0.2))
result['alpha']=alpha
#alpha 0.1 to 0.5 gives maximum number of russell conjugates
result.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Sentiment classification/Final version_pw/result_disciplinarian_upto_5.csv')

#Lets try these alpha on another word like "lazy"

result=pd.DataFrame()
alpha=[]
for i in np.arange(0.1,5,0.2):
    df=[]
    w2 =(model['lazy'])-(i*l)
    x= model.most_similar(positive=[w2])
    df.append(x)
    result= result.append(df)

alpha=np.array(np.arange(0.1,5,0.2))
result['alpha']=alpha
#alpha 0.1 to 0.5 gives maximum number of russell conjugates
result.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Sentiment classification/Final version_pw/result_lazy_upto_5.csv')


#Lets try these alpha on another word like "liberated"

result=pd.DataFrame()
alpha=[]
for i in np.arange(0.1,5,0.2):
    df=[]
    w2 =(model['liberated'])-(i*l)
    x= model.most_similar(positive=[w2])
    df.append(x)
    result= result.append(df)

alpha=np.array(np.arange(0.1,5,0.2))
result['alpha']=alpha
#None qualify
result.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Sentiment classification/Final version_pw/result_liberated_upto_5.csv')


#Lets try alpha=0.1 to multiple positive words:

data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/pw/data1.csv',header=None)

result=pd.DataFrame()
alpha=[]
word=[]
for index, row in data1.iterrows():
    df=[]
    word.append(row[0])
    w2 =(model[row[0]])-(0.1*l)
    x= model.most_similar(positive=[w2])
    df.append(x)
    result= result.append(df)

result['word']=word
result.to_csv('F:/somebody gonna get hurt/Github/Model/Models/Sentiment classification/Final version_pw/result_words_alpha_point1.csv')

