'''
Created on 25-Aug-2021

@author: somsh
'''
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf

def Gaussian(p):
    f = 1
    sigma=1/f
    y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(p**2)/(2*(sigma**2)))
    return y    
                                
def Gaussian_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    if in_channels != 1:
        msg = "GaussConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    a_ = np.linspace(1, 10, out_channels).reshape(-1, 1)
    b_ = np.linspace(0, 10, out_channels).reshape(-1, 1)
    time_disc = np.linspace(-8,8, int(kernel_size))
    p = time_disc - b_/a_
    Gaussian_filter = Gaussian(p)
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Gaussian_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Gaussian_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Gaussian_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("Gaussian filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("Gaussian(t)")
        plt.savefig("Gaussian_filter/MotherGaussian"+str(i)+".png")
        plt.close()
    Gaussian_filter = Gaussian_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Gaussian_filter),dtype)
    return filter

