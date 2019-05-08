# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:59:45 2019

@author: pragy
"""

import numpy as np
import pandas as pd
import seaborn as sns

range= np.max(nonRC_dist) - np.min(nonRC_dist)
width = range/18

l=[]
l.append(np.min(nonRC_dist))
a=np.min(nonRC_dist)
x=0
while x<19:
    a=a+width
    l.append(a)
    x=x+1
    

RC_sorted= RC_dist.sort_values()
type(RC_sorted)
len(l)
num=[]
rt=1
while rt<20:
    count=0
    for w in RC_sorted:
        if w<l[rt] and w>=l[rt-1]:
            count=count+1
    num.append(count)
    rt=rt+1


nonRC_sorted= nonRC_dist.sort_values()
type(nonRC_sorted)
len(l)
num_nonRC=[]
rt=1
while rt<20:
    count=0
    for w in nonRC_sorted:
        if w<l[rt] and w>=l[rt-1]:
            count=count+1
    num_nonRC.append(count)
    rt=rt+1
    
divder_RC=sum(num)
divider_nonRC=sum(num_nonRC)

percentages_RC=[x*100/divder_RC for x in num]
percentages_nonRC= [x*100/divider_nonRC for x in num_nonRC]

len(percentages_RC)
percentages_RC.append(0)
len(percentages_nonRC)
percentages_nonRC.append(0)

l= [round(x,3) for x in l]
dff= pd.DataFrame(
    {'bins': l,
     'percent_RC':percentages_RC,
     'percent_nonRC': percentages_nonRC
    })


df = pd.DataFrame(dict(x=np.random.poisson(4, 500)))
ax = sns.barplot(x='bins',y='percent_RC', data=dff)
plt.xticks(rotation=70)
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 20

RC_dist.to_csv('F:/somebody gonna get hurt/Github/Model/RC_dist.csv')
nonRC_dist.to_csv('F:/somebody gonna get hurt/Github/Model/nonRC_dist.csv')
