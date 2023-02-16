'''
Created on 08-Aug-2021

@author: somsh
'''
import numpy as np
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf
import os

def Daubechies1_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db1').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies1_filter")
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
        plt.title("Daubechies1_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies1_filter/MotherDaubechies1"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies2_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db2').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies2_filter")
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
        plt.title("Daubechies2_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies2_filter/MotherDaubechies2"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies3_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db3').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies3_filter")
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
        plt.title("Daubechies3_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies3_filter/MotherDaubechies3"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies4_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db4').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies4_filter")
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
        plt.title("Daubechies4_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies4_filter/MotherDaubechies4"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies5_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db5').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies5_filter")
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
        plt.title("Daubechies5_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies5_filter/MotherDaubechies5"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies6_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db6').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies6_filter")
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
        plt.title("Daubechies6_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies6_filter/MotherDaubechies6"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies7_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db7').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies7_filter")
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
        plt.title("Daubechies7_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies7_filter/MotherDaubechies7"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies8_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db8').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies8_filter")
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
        plt.title("Daubechies8_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies8_filter/MotherDaubechies8"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies9_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db9').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies9_filter")
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
        plt.title("Daubechies9_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies9_filter/MotherDaubechies9"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

def Daubechies10_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    level = 4
    a_ = np.linspace(1, 10, out_channels)
    b_ = np.linspace(0, 10, out_channels)
    phase = b_/a_
    wavelet = pywt.Wavelet('db10').wavefun(level)
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
        msg = "DaubechiesConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    Daubechies_filter = np.array(Daubechies_filter) 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Daubechies10_filter")
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
        plt.title("Daubechies10_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Daubechies10_filter/MotherDaubechies10"+str(i)+".png")
        plt.close()
    Daubechies_filter = Daubechies_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Daubechies_filter),dtype)
    return filter

