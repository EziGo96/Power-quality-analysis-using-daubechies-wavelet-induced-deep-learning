'''
Created on 25-Aug-2021

@author: somsh
'''

import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf

def Mexh(p):
    f=1
    sigma=1/f
    y = (2/(np.sqrt(3*sigma)*np.pi**(1/4)))*(1-(p/sigma)**2)*np.exp(-(p**2)/(2*(sigma**2)))
    return y
   
def Mexh_filter(shape,dtype=tf.float32):
    out_channels = shape[2]
    in_channels = shape[1]
    kernel_size = shape[0]
    if in_channels != 1:
        msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        raise ValueError(msg)
    a_ = np.linspace(1, 10, out_channels).reshape(-1, 1)
    b_ = np.linspace(0, 10, out_channels).reshape(-1, 1)
    time_disc = np.linspace(-8,8, int(kernel_size))
    p = time_disc - b_/a_
    Mexh_filter = Mexh(p)
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Mexh_filter")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(Mexh_filter.shape[0]):
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(Mexh_filter[i])
        plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        plt.title("MexHat filter"+str(i))
        # plt.xlabel("t")
        # plt.ylabel("MexH(t)")
        plt.savefig("Mexh_filter/MotherMexH"+str(i)+".png")
        plt.close()
    Mexh_filter = Mexh_filter.T.reshape(kernel_size,in_channels,out_channels)
    filter = tf.dtypes.cast(tf.convert_to_tensor(Mexh_filter),dtype)
    return filter

# class Mexh_filter(tf.keras.initializers.Initializer):
#     def __init__(self, inputSignals):
#         self.inputSignals = inputSignals
#
#     def __call__(self, shape, dtype=None):
#         out_channels = shape[2]
#         in_channels = shape[1]
#         kernel_size = shape[0]
#         if kernel_size % 2 != 0:
#             kernel_size = kernel_size - 1
#         if in_channels != 1:
#             msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
#             raise ValueError(msg)
#         a_ = np.linspace(1, 10, out_channels).reshape(-1, 1)
#         b_ = np.linspace(0, 10, out_channels).reshape(-1, 1)
#         time_disc_right = np.linspace(0, (kernel_size / 2)-1, int(kernel_size / 2))
#         time_disc_left = np.linspace(-(kernel_size / 2)+1, -1, int(kernel_size / 2))
#         p1 = time_disc_right - b_/a_
#         p2 = time_disc_left - b_/a_
#         p = np.concatenate((p2, p1), axis = 1)
#         filters = []
#         for i in range(self.inputSignals):
#             Mexh_filter = Mexh(p,self.inputSignals[i])
#             current_directory = os.getcwd()
#             final_directory = os.path.join(current_directory, "Mexh_filter",str(i))
#             if not os.path.exists(final_directory):
#                 os.makedirs(final_directory)
#             for i in range(Mexh_filter.shape[0]):
#                 plt.style.use("ggplot")
#                 plt.figure()
#                 plt.plot(Mexh_filter[i])
#                 plt.title("MexHat filter"+str(i))
#                 plt.xlabel("t")
#                 plt.ylabel("MexH(t)")
#                 plt.savefig("Mexh_filter/MotherMexH"+str(i)+".png")
#             Mexh_filter = Mexh_filter.T.reshape(kernel_size,in_channels,out_channels)
#             filters.append(tf.dtypes.cast(tf.convert_to_tensor(Mexh_filter),dtype))
#             return filters
#
#     def get_config(self):  # To support serialization
#         return {'inputSignal': self.inputSignal}

