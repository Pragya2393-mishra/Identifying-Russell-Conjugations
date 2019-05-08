# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:39:27 2019

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

#good, evil
l= model['good'] - model['evil']
w2 =model['leader']-l
model.most_similar(negative=[w2])
#    [('good', 0.46881693601608276),
# ('nice', 0.3954629898071289),
# ('terrific', 0.36072060465812683),
# ('decent', 0.34660932421684265),
# ('excellent', 0.3443392515182495),
# ('excelent', 0.31148988008499146),
# ('better', 0.30424726009368896),
# ('fantastic', 0.29672253131866455),
# ('solid', 0.2935830354690552),
# ('great', 0.28931963443756104)]

l= model['good'] - model['bad']
w2 =model['leader']-l
model.most_similar(positive=[w2])
#[('leader', 0.7792259454727173),
# ('Leader', 0.44592583179473877),
# ('leaders', 0.418234646320343),
# ('chieftain', 0.3900662660598755),
# ('SIR_Menzies_Campbell', 0.38821831345558167),
# ('supremo_HD_Devegowda', 0.38694968819618225),
# ('president', 0.3858824372291565),
# ('leader_Ali_Ahmeti', 0.3819548189640045),
# ('leadership', 0.3776627779006958),
# ('minister_Sunil_Deshmukh', 0.376039981842041)]

l= model['obstinate'] - model['pigheaded']
w2 =model['leader']-l
model.most_similar(positive=[w2])
#[('leader', 0.7205183506011963),
# ('premier', 0.42264223098754883),
# ('leadership', 0.4149612784385681),
# ('pigheaded', 0.38719725608825684),
# ('pioneer', 0.3864291310310364),
# ('president', 0.3726041913032532),
# ('titan', 0.36822691559791565),
# ('frontrunner', 0.36071935296058655),
# ('Djindjic_Serbia', 0.35599297285079956),
# ('chairman', 0.3552280068397522)]

l= model['pigheaded'] - model['firm']
w2 =model['leader']-l
model.most_similar(positive=[w2])
#[('firm', 0.617813229560852),
# ('leader', 0.6073501706123352),
# ('consulting_firm', 0.441483736038208),
# ('consultancy', 0.4281071424484253),
# ('NNIT', 0.3929162621498108),
# ('Digico', 0.39123672246932983),
# ('Plasmon_PLC', 0.38636085391044617),
# ('Yudelson_Associates', 0.3779566287994385),
# ('Contrack_International', 0.37749359011650085),
# ('Incepta', 0.37698838114738464)]

l= model['liberated'] - model['strange']
w2 =model['leader']-l
model.most_similar(negative=[w2])
#[('liberated', 0.4743567705154419),
# ('liberate', 0.304787278175354),
# ('Liberated', 0.2805936932563782),
# ('demobilized', 0.27725738286972046),
# ('By_NIKI_DOYLE', 0.24869310855865479),
# ('reequipped', 0.24767480790615082),
# ('Thousand_Oaks_CA_PRWEB', 0.24745218455791473),
# ('RSAs', 0.23861439526081085),
# ('----------------------------------------------------------_tons',
#  0.23825335502624512),
# ('liberation', 0.23602023720741272)]

l= model['good'] - model['bad']
w2 =(model['leader'])-(10*l)
model.most_similar(positive=[w2])
#[('bad', 0.3742288053035736),
# ('maniacal_killer', 0.3708759546279907),
# ('insuring_repackaged_subprime_mortgages', 0.35758519172668457),
# ('Bad', 0.3409612774848938),
# ('subprime_mortgage_debacle', 0.3343667984008789),
# ('become_dehydrated_taters', 0.32744693756103516),
# ('anti_semitic_tirade', 0.32215815782546997),
# ('flimflams', 0.31869199872016907),
# ('hedge_fund_blowups', 0.3183669447898865),
# ('Um_Mazin', 0.3176092207431793)]


l= model['good'] - model['bad']
w2 =(model['leader'])-(10*l)
model.most_similar(negative=[w2])
#[('excellent', 0.45849770307540894),
# ('great', 0.4349369406700134),
# ('terrific', 0.4085104465484619),
# ('nice', 0.3826293349266052),
# ('fantastic', 0.3728930950164795),
# ('good', 0.3687405586242676),
# ('decent', 0.3664875030517578),
# ('wonderful', 0.36336466670036316),
# ('solid', 0.3529513478279114),
# ('tremendous', 0.3439888656139374)]

l= model['good'] - model['bad']
w2 =(10*model['leader'])-(l)
model.most_similar(negative=[w2])
#[('Related_KOMU_Stories', 0.3097790479660034),
# ('By_LINDA_WALTON', 0.2725362181663513),
# ('By_NIKI_DOYLE', 0.2552405595779419),
# ('www.crowncenter.com_###-###-####', 0.25145214796066284),
# ('By_Settu_Shankar', 0.25058454275131226),
# ('adobe_dreamweaver_download', 0.24170684814453125),
# ('Tammy_Dombeck', 0.2409517765045166),
# ('Coronet', 0.23796729743480682),
# ('Blocked_Amount_ASBA', 0.23713327944278717),
# ('By_MARLISA_KEYES', 0.23691534996032715)]

l= model['good'] - model['bad']
w2 =(10*model['leader'])-(l)
model.most_similar(positive=[w2])
#[('leader', 0.9971873760223389),
# ('leadership', 0.556816577911377),
# ('Leader', 0.5510758757591248),
# ('leaders', 0.545345664024353),
# ('premier', 0.5005115270614624),
# ('president', 0.4579772651195526),
# ('leading', 0.44386160373687744),
# ('parliamentarian', 0.44270843267440796),
# ('standard_bearer', 0.4389641284942627),
# ('SIR_Menzies_Campbell', 0.42823314666748047)]

l= model['good'] - model['bad']
w2 =(2*model['leader'])-(l)
model.most_similar(positive=[w2])
#[('leader', 0.9325749278068542),
# ('Leader', 0.5232031345367432),
# ('leaders', 0.5059598088264465),
# ('leadership', 0.49132734537124634),
# ('premier', 0.45195960998535156),
# ('president', 0.44264182448387146),
# ('SIR_Menzies_Campbell', 0.4279201924800873),
# ('chieftain', 0.42129212617874146),
# ('parliamentarian', 0.41963082551956177),
# ('minister_Sunil_Deshmukh', 0.40881723165512085)]

l= model['king'] - model['queen']
w2 =(model['man'])-(l)
model.most_similar(positive=[w2])
#[('woman', 0.760943591594696),
# ('man', 0.7474270462989807),
# ('girl', 0.6139994263648987),
# ('teenage_girl', 0.6040961742401123),
# ('teenager', 0.5825759172439575),
# ('lady', 0.5752554535865784),
# ('boy', 0.5077577233314514),
# ('policewoman', 0.5066847801208496),
# ('schoolgirl', 0.5052095651626587),
# ('blonde', 0.48696184158325195)]