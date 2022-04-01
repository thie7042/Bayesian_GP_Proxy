import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import tensorflow as tf

print('test')

filename = 'finalized_model.sav'

my_model = pickle.load(open(filename, 'rb'))

print(my_model)
print(type(my_model))

print('_________________________________')



from sklearn.model_selection import train_test_split

from numpy import array




print('________')





###################


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


x1_values = np.linspace(-5, 10, 100)
x2_values = np.linspace(0, 15, 100)

x_ax, y_ax = np.meshgrid(x1_values, x2_values)
x_ax = x1_values
y_ax = x2_values



x_ax = NormalizeData(x1_values)
y_ax = NormalizeData(x2_values)


c = 2
r = 100
T1 = [[0] * c for i in range(r)] # loop will run for the length of the outer list

for i in range(r):
    T1[i][0] = x_ax[i]
    T1[i][1] = y_ax[i]


print(T1)

print("_________")


T1 = array(T1)

predicted_values = my_model.predict(T1)


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

z_val = predicted_values[0]



z_val = z_val.tolist()
z_val = [item for sublist in z_val for item in sublist]

z_val = NormalizeData(z_val)


print("_________")


print(z_val)



fig, ax = plt.subplots()
ax = plt.axes(projection='3d')
plt.plot(x_ax,y_ax,z_val,'o', color='red')




## Getting obj to check

from skopt.benchmarks import branin as branin


import numpy as np


from skopt.benchmarks import branin as branin

from trieste.acquisition.rule import OBJECTIVE
from matplotlib import pyplot as plt





# Define the benchmark function


x_ax, y_ax = np.meshgrid(x1_values, x2_values)

vals = np.c_[x_ax.ravel(), y_ax.ravel()]


fx = np.reshape([branin(val) for val in vals], (100, 100))


x_ax = NormalizeData(x_ax)
y_ax = NormalizeData(y_ax)

fx = NormalizeData(fx)

ax.plot_surface(x_ax, y_ax, fx, cmap='jet')

plt.show()



##


