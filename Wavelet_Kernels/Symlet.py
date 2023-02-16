'''
Created on 12-Aug-2021

@author: somsh
'''
import numpy as np
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf
import os



def Symlet2_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym2').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet2_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet2_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet2_filter/MotherSymlet2"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Symlet3_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym3').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet3_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet3_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet3_filter/MotherSymlet3"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Symlet4_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym4').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet4_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet4_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet4_filter/MotherSymlet4"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Symlet5_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym5').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet5_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet5_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet5_filter/MotherSymlet5"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Symlet6_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym6').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet6_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet6_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet6_filter/MotherSymlet6"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Symlet7_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym7').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet7_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet7_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet7_filter/MotherSymlet2"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Symlet8_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym8').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet8_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet8_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet8_filter/MotherSymlet8"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Symlet9_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym9').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet9_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet9_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet9_filter/MotherSymlet9"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Symlet10_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('sym10').wavefun(level)
    num_sample=len(wavelet[2])
    phase = (phase*num_sample/kernel_size).astype(int)
    psi = wavelet[1]
    psi_l = []
    for i in range(len(phase)):
        l = psi.tolist()
        for j in range(phase[i]):
            l.pop(-1)
            l.insert(0,0.0)
        psi_l.append(l)
    Daubechies_filter=[]
    for i in psi_l:
        idx = np.round(np.linspace(0, len(i) - 1,kernel_size)).astype(int)
        j = [i[_] for _ in idx]
        Daubechies_filter.append(j)
    if in_channels != 1:
        msg = "SymletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Symlet10_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Daubechies_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Daubechies_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Symlet10_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Symlet10_filter/MotherSymlet10"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

