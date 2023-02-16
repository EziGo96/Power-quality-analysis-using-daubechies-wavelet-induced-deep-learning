'''
Created on 03-Aug-2021

@author: somsh
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import Model.config as config
import os
Class_list=os.listdir(config.ORIG_INPUT_DATASET)
n=len(Class_list)
ext=Class_list[0][-13:]
for i in range(n):
    Class_list[i]=Class_list[i][:-13]
print("Class List:",Class_list)
print("no. of Classes:",n)
choice="Y"
while choice=="Y":
    index=int(input("Enter index of Class from Class List: "))
    # Class="A"
    pathname=os.path.join(config.ORIG_INPUT_DATASET,Class_list[index]+ext)
    print(pathname)
    df=pd.read_excel(pathname,header=None)
    index=int(input("Enter Signal index: "))
    A=np.array(df.iloc[index])
    N=len(A)
    plt.figure()
    plt.plot(A)
    A_F=fft(A)
    realA_F=[i.real for i in A_F]
    imagA_F=[i.imag for i in A_F]
    plt.figure()
    plt.plot(realA_F,imagA_F,color="red")
    plt.figure()
    plt.polar(np.angle(A_F),np.abs(A_F))
    A_F=A_F[:len(A_F)//2]
    i=np.argmax(np.abs(A_F))
    print(i)
    choice=input("continue? Y/N: ").upper()
    plt.figure()
    plt.plot(np.abs(A_F))
    # plt.figure()
    # plt.plot(np.angle(A_F))
    plt.show()


