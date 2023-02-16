'''
Created on 07-Apr-2020

@author: somsh
'''
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from Model import config
import pandas as pd
from sklearn.model_selection import train_test_split
from Wavelet_Kernels.MexicanHat import Mexh_filter
from Wavelet_Kernels.Morlet import Morlet_filter
from Wavelet_Kernels.Laplace import Laplace_filter
from Wavelet_Kernels.Gauss import Gaussian_filter
from Wavelet_Kernels.Daubechies import Daubechies1_filter,Daubechies2_filter,Daubechies3_filter,Daubechies4_filter,Daubechies5_filter, Daubechies6_filter, Daubechies7_filter, Daubechies8_filter, Daubechies9_filter,Daubechies10_filter
from Wavelet_Kernels.Symlet import Symlet2_filter,Symlet3_filter,Symlet4_filter,Symlet5_filter,Symlet6_filter,Symlet7_filter,Symlet8_filter,Symlet9_filter,Symlet10_filter
from tensorflow.keras.models import model_from_json
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
from pandas import DataFrame as df

BS = 20
Classes=os.listdir(config.ORIG_INPUT_DATASET)
n=len(Classes)
ext=Classes[0][-13:]
for i in range(n):
    Classes[i]=Classes[i][:-13]
print("Classes:",Classes)
print("no. of Classes:",n)
X={}
for i in Classes:
    X[i]=pd.read_excel(os.path.join(config.ORIG_INPUT_DATASET,i+ext),header=None).values
Y={}
for i in Classes:
    Y[i]=[Classes.index(i) for _ in X[i]]
    
X_train={}
X_test={}
Y_train={}
Y_test={}
for j,k in zip(X.keys(),Y.keys()):
    X_train[j], X_test[j], Y_train[k], Y_test[k] = train_test_split(X[j], Y[k], train_size=config.TRAIN_SPLIT, random_state=0)
    
arrays=[X_test[_] for _ in X_test.keys()]
X_test_DS=np.concatenate(arrays, axis=0).astype('float32')
X_test_DS=np.reshape(X_test_DS,(X_test_DS.shape[0],X_test_DS.shape[1],1))
arrays=[Y_test[_] for _ in Y_test.keys()]
Y_test_DS=np.concatenate(arrays, axis=0)
Y_test_DS=np.reshape(Y_test_DS,(Y_test_DS.shape[0],1))

model=load_model('model.hp5',custom_objects={'Daubechies4_filter': Daubechies4_filter,'Daubechies5_filter': Daubechies5_filter, 'tf': tf})
json_file = open('architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json, custom_objects={'Daubechies4_filter': Daubechies4_filter,'Daubechies5_filter': Daubechies5_filter, 'tf': tf})
model1.load_weights("model_wts.hdf5")
# Testing
totalTest = len(list(X_test_DS))
# initialize the testing generator
# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
predIdxs = model.predict(X_test_DS,batch_size = BS)
predIdxs1 = model1.predict(X_test_DS,batch_size = BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
predIdxs1 = np.argmax(predIdxs1, axis=1)

conf_mat = confusion_matrix(Y_test_DS, predIdxs)
conf_mat1 = confusion_matrix(Y_test_DS, predIdxs1)
# show a confusion matrix and formatted classification report

f=open("result.txt","w")
f.write(str(conf_mat))
f.write(classification_report(Y_test_DS, predIdxs))
f.write("\n")
f.write(str(conf_mat1))
f.write(classification_report(Y_test_DS, predIdxs1))
f.close()

df_cm = df(conf_mat, index=Classes, columns=Classes)
df_cm1 = df(conf_mat1, index=Classes, columns=Classes)

pretty_plot_confusion_matrix(df_cm)
pretty_plot_confusion_matrix(df_cm1)

print("[INFO] Test finished")