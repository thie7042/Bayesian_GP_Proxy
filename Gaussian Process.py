import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import plotly.express as px
import cufflinks as cf
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf


from skopt.benchmarks import branin as branin
from skopt.benchmarks import hart6 as hart6_

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(1793)
tf.random.set_seed(1793)





from trieste.objectives import (
    scaled_branin,
    SCALED_BRANIN_MINIMUM,
    BRANIN_SEARCH_SPACE,
)
from trieste.objectives.utils import mk_observer

from trieste.space import Box





search_space = BRANIN_SEARCH_SPACE  # predefined search space, for convenience
search_space = Box([0, 0], [1, 1])  # define the search space directly



##############
def plot_branin():
    fig, ax = plt.subplots()
    #fig, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 8))

    x1_values = np.linspace(-5, 10, 100)
    x2_values = np.linspace(0, 15, 100)
    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([branin(val) for val in vals], (100, 100))

    cm = ax.pcolormesh(x_ax, y_ax, fx,
                       norm=LogNorm(vmin=fx.min(),
                                    vmax=fx.max()),
                       cmap='viridis_r')

    minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
    ax.plot(minima[:, 0], minima[:, 1], "r.", markersize=14,
            lw=0, label="Minima")

    cb = fig.colorbar(cm)
    cb.set_label("f(x)")

    ax.legend(loc="best", numpoints=1)

    ax.set_xlabel("$X_0$")
    ax.set_xlim([-5, 10])
    ax.set_ylabel("$X_1$")
    ax.set_ylim([0, 15])
    plt.show()

    ax1 = plt.axes(projection='3d')
    ax1.plot_surface(x_ax, y_ax, fx, cmap='jet')


    plt.show()



plot_branin()
plt.show()

#######



"""fig = plot_function_plotly(
    scaled_branin, search_space.lower, search_space.upper, grid_density=20
)
fig.update_layout(height=400, width=400)
fig.show()"""

print("______________________________________________________________")
print("Hello world")