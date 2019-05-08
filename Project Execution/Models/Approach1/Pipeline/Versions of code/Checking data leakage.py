# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:33:49 2019

@author: pragy
"""

import numpy as np
import pandas as pd
###############################################################################################
data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19_v3.csv')
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files/test18.csv')
data2 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27_v3.csv')

data1=data1.drop('y',axis=1)
test1=test1.drop('y',axis=1)
data2=data2.drop('y',axis=1)

df = pd.merge(data1, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
len(df[df.Exist==True])

df = pd.merge(data2, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
leak = df[df.Exist==True]
len(leak)

#############################################################################################
data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19_v2.csv')
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files/test17.csv')
data2 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27_v2.csv')

data1=data1.drop('y',axis=1)
test1=test1.drop('y',axis=1)
data2=data2.drop('y',axis=1)

df = pd.merge(data1, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
len(df[df.Exist==True])

df = pd.merge(data2, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
leak = df[df.Exist==True]
len(leak)

#############################################################################################
data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19_v2.csv')
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files/test17.csv')
data2 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27.csv')

data1=data1.drop('y',axis=1)
test1=test1.drop('y',axis=1)
data2=data2.drop('y',axis=1)

df = pd.merge(data1, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
len(df[df.Exist==True])

df = pd.merge(data2, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
leak = df[df.Exist==True]
len(leak)
#############################################################################################

data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19.csv')
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files/test9.csv')
data2 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27_v3.csv')

data1=data1.drop('y',axis=1)
test1=test1.drop('y',axis=1)
data2=data2.drop('y',axis=1)

df = pd.merge(data1, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
len(df[df.Exist==True])

df = pd.merge(data2, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
leak = df[df.Exist==True]
len(leak)
#############################################################################################

data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19_v3.csv')
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files/test9_20.csv')
data2 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27_v3.csv')

data1=data1.drop('y',axis=1)
test1=test1.drop('y',axis=1)
data2=data2.drop('y',axis=1)

df = pd.merge(data1, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
len(df[df.Exist==True])

df = pd.merge(data2, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
leak = df[df.Exist==True]
len(leak)
#############################################################################################

data1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav19.csv')
test1 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/Test files/test9_20.csv')
data2 = pd.read_csv('F:/somebody gonna get hurt/Github/Model/data/datav27_v3.csv')

data1=data1.drop('y',axis=1)
test1=test1.drop('y',axis=1)
data2=data2.drop('y',axis=1)

df = pd.merge(data1, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
len(df[df.Exist==True])

df = pd.merge(data2, test1, on=['x1','x2'], how='left', indicator='Exist')
df['Exist'] = np.where(df.Exist == 'both', True, False)
leak = df[df.Exist==True]
len(leak)
#############################################################################################