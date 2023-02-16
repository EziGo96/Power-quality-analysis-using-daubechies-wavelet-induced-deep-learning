'''
Created on 25-Aug-2021

@author: somsh
'''
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf

def Morlet(p): 
    f=1
    sigma=1/f
    C_sigma=(1+np.exp(-(sigma**2))-2*np.exp(-(3/4)*(sigma**2)))**(-1/2)
    y = C_sigma*(np.pi**(-1/4))*np.exp(-(p**2)/(2*(sigma**2)))*np.cos(2*np.pi*p/sigma)
    return y

def Morlet_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    if in_channels != 1:
        msg = "MorletConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    a_ = np.linspace(1, 10, out_channels).reshape(-1, 1)
    b_ = np.linspace(0, 10, out_channels).reshape(-1, 1)
    time_disc = np.linspace(-8,8, int(kernel_size))
    p = time_disc - b_/a_
    Morlet_filter = Morlet(p)
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Morlet_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Morlet_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Morlet_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        # plt.title("Morlet filter"+str(i))
        # plt.xlabel("t")
        plt.ylabel("Morlet(t)")
        plt.savefig("Morlet_filter/MotherMorlet"+str(i)+".png")
        plt.close()
    Morlet_filter = Morlet_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Morlet_filter),dtype)
    return filter

