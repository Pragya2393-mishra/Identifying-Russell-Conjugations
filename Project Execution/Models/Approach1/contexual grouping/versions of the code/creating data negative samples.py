# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:24 2019

@author: pragy
"""

import os
import gensim
import numpy as np
import pandas as pd
from random import shuffle

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