'''
Created on 25-Aug-2021

@author: somsh
'''
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf

def Laplace(p):
    a = 0.08
    prop= 1
    ep = 0.03
    tau = 0.0
    f = 1
    w = 2 * np.pi * f
    q = (1 - ep**2)
    y = a * np.exp((-ep / (np.sqrt(q))) * (w * (p - tau)/a)) * (np.sin(w * (p - tau)/a))
    return y

def Laplace_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    if in_channels != 1:
        msg = "LaplaceConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    a_ = np.linspace(1, 10, out_channels).reshape(-1, 1)
    b_ = np.linspace(0, 10, out_channels).reshape(-1, 1)
    time_disc = np.linspace(0,16, int(kernel_size))
    p = time_disc - b_/a_
    Laplace_filter = Laplace(p)
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Laplace_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Laplace_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Laplace_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Laplace_filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Laplace(t)")
        plt.savefig("Laplace_filter/MotherLaplace"+str(i)+".png")
        plt.close()
    Laplace_filter = Laplace_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Laplace_filter),dtype)
    return filter  

