'''
Created on 23-Jul-2021

@author: somsh
'''
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
from pandas import DataFrame as df
import numpy as np
import os
from Model import config 
Classes=os.listdir(config.ORIG_INPUT_DATASET)
n=len(Classes)
ext=Classes[0][-13:]
for i in range(n):
    Classes[i]=Classes[i][:-13]
print("Classes:",Classes)
print("no. of Classes:",n)
array = np.array([[100  , 0  , 0 ,  0  , 0  , 0,   0 ,  0 ,  0   ,0 , 0 ,  0 ,  0]
 ,[  0 , 98 ,  0  , 2,   0,   0,   0,   0,   0,   0,   0,   0 ,  0]
, [  0  , 0,  94,   6 ,  0,   0,   0,   0,   0,   0,   0,   0,   0]
, [  0  , 0,   0 ,100  , 0,   0,   0,   0,   0,   0,   0,   0,   0]
, [  0  , 0 ,  0  , 0 ,100,   0,   0,   0,   0,   0,   0,   0,   0]
, [  0  , 0  , 0   ,0  , 0, 100,   0,   0,   0,   0,   0,   0,   0]
, [  0 ,  0   ,0   ,0   ,1,   0,  98,   0,   0,   1,   0,   0,   0]
, [  0 ,  0   ,0   ,0   ,0,   0,   0,  98,   0,   0,   2,   0,   0]
, [  0  , 0   ,0   ,0   ,0,   0,   0,   0, 100,   0,   0,   0,   0]
, [  0   ,0   ,0   ,0   ,6,   0,   0,   0,   0,  94,   0,   0,   0]
, [  0   ,0   ,0   ,0   ,0,   0,   0,   0,   1,   0,  99,   0,   0]
, [  0   ,0   ,0   ,0   ,7,   0,   0,   0,   0,   0,   0,  93,   0]
, [  0   ,0   ,0   ,0   ,0,   0,   0,   0,   1,   0,   5,   0,  94]])
#get pandas dataframe
df_cm = df(array, index=Classes, columns=Classes)
#colormap: see this and choose your more dear
pretty_plot_confusion_matrix(df_cm)