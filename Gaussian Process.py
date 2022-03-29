import numpy as np
import tensorflow as tf
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow

from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.bayesian_optimizer import Record
from trieste.data import Dataset
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.objectives import scaled_branin, SCALED_BRANIN_MINIMUM
from trieste.objectives.utils import mk_observer
from trieste.space import Box


from trieste.models.gpflow import GaussianProcessRegression


from skopt.benchmarks import branin as branin


from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from trieste.space import Box

search_space = Box([0, 0], [1, 1])  # define the search space directly. We are looking in a 2d plane (2-dimensonal, 2 valriables ranging 0-1)





np.random.seed(1793)
tf.random.set_seed(1793)





from trieste.objectives import (
    scaled_branin,
    SCALED_BRANIN_MINIMUM,
    BRANIN_SEARCH_SPACE,
)



############## Plotting of benchmark function. This is just for context ##########
def plot_branin():

    fig, ax = plt.subplots()
    #fig, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 8))

    x1_values = np.linspace(-5, 10, 100)
    x2_values = np.linspace(0, 15, 100)
    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]

    # Define the benchmark function
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


###########


import trieste
from trieste.models.gpflow import build_gpr



#TEST#\/\/\/\/\/
def build_model(data, kernel_func=None):
    """kernel_func should be a function that takes variance as a single input parameter"""
    variance = tf.math.reduce_variance(data.observations)
    if kernel_func is None:
        kernel = gpflow.kernels.Matern52(variance=variance)
    else:
        kernel = kernel_func(variance)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)

# Number of preliminary data points that we need to sample
num_initial_points = 5

# This is using the Sobol Sampling technique
initial_query_points = search_space.sample_sobol(num_initial_points)


x_1_sampling = []
x_2_sampling = []

for i in range(initial_query_points.shape[0]):
    x_1_sampling.append(initial_query_points[i][0])
    x_2_sampling.append(initial_query_points[i][1])

print(initial_query_points)
plt.plot(x_1_sampling,x_2_sampling,'o', color='black')
plt.show()

########### ___ Initial data (This should be gethered from FEM ___ ###########
observer = trieste.objectives.utils.mk_observer(scaled_branin)
initial_data = observer(initial_query_points)
########### ___ Initial data (This should be gethered from FEM ___ ###########


n_steps = 5
model = build_model(initial_data)
ask_tell = AskTellOptimizer(search_space, initial_data, model)
for step in range(n_steps):
    print(f"Ask Trieste for configuration #{step}")
    new_config = ask_tell.ask()

    print("Saving Trieste state to re-use later")
    state: Record[None] = ask_tell.to_record()
    saved_state = pickle.dumps(state)

    print(f"In the lab running the experiment #{step}.")
    new_datapoint = scaled_branin(new_config)

    print("Back from the lab")
    print("Restore optimizer from the saved state")
    loaded_state = pickle.loads(saved_state)
    ask_tell = AskTellOptimizer.from_record(loaded_state, search_space)
    ask_tell.tell(Dataset(new_config, new_datapoint))


####^^^^^^^



print("______________________________________________________________")
print("Hello world! Let's Create our Proxy!")



gpflow_model = build_gpr(initial_data, search_space, likelihood_variance=1e-7)
model = GaussianProcessRegression(gpflow_model, num_kernel_samples=100)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 15
result = bo.optimize(num_steps, initial_data, model)
dataset = result.try_get_final_dataset()

print("______________________________________________________________")


query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")


print("___________________________VISUALIZE 2D___________________________________")


x_pos_best = query_points[arg_min_idx, :][0]
y_pos_best = query_points[arg_min_idx, :][1]
z_pos_best = observations[arg_min_idx, :][0]


print(query_points)
print(observations)

x_1 = []
x_2 = []
obs = []



for i in range(query_points.shape[0]):
    x_1.append(query_points[i][0])
    x_2.append(query_points[i][1])
    obs.append(observations[i][0])

print(query_points)
print(x_1)
print(x_2)

fig, ax = plt.subplots()
plt.plot(x_1,x_2,'o', color='black')
plt.plot(x_pos_best, y_pos_best,'o', color='red')

#plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)
ax.set_xlabel("$X_0$")
ax.set_xlim([0, 1])
ax.set_ylabel("$X_1$")
ax.set_ylim([0, 1])


#Contour

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



x1_values = np.linspace(-5, 10, 100)
x2_values = np.linspace(0, 15, 100)
x_ax, y_ax = np.meshgrid(x1_values, x2_values)
vals = np.c_[x_ax.ravel(), y_ax.ravel()]
fx = np.reshape([branin(val) for val in vals], (100, 100))

fx = NormalizeData(fx)
x_ax = NormalizeData(x1_values)
y_ax = NormalizeData(x2_values)


levels = np.arange(0,1,0.01)
ax.contour(x_ax, y_ax, fx,levels=levels)
#cb = fig.colorbar(cm)


plt.show()


print("___________________________VISUALIZE 3D___________________________________")



obs = NormalizeData(obs)



ax1 = plt.axes(projection='3d')
levels = np.arange(0,1,0.01)
ax1.contour(x_ax, y_ax, fx,levels=levels)

plt.plot(x_1,x_2,obs,'o', color='black')
plt.plot(x_pos_best, y_pos_best, z_pos_best,'o', color='red')

plt.show()


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
