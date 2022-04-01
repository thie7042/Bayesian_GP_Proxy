
import util
from util.inequality_constraints_utils import plot_objective_and_constraints
import numpy as np
import tensorflow as tf
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset

import trieste
from trieste.space import Box

np.random.seed(1799)
tf.random.set_seed(1799)



###########################
# Class is a blueprint #
##########################
class Sim:

    #The constraint is satisfied when constraint(input_data) <= threshold.
    threshold = 0.5

    # Static methods only need to use the parameters they get passed
    # This is our objective function
    @staticmethod
    def objective(input_data):
        # X = all second last entries in input_data, Y = all last entries
        x, y = input_data[..., -2], input_data[..., -1]
        z = tf.cos(2.0 * x) * tf.cos(y) + tf.sin(x)

        print(z[:, None])
        print("Test :)")

        return z[:, None]

    @staticmethod
    def constraint(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return z[:, None]


search_space = Box([0, 0], [6, 6])


OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"



def observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, Sim.objective(query_points)),
        CONSTRAINT: Dataset(query_points, Sim.constraint(query_points)),
    }


#
num_initial_points = 5

# Collecting preliminary sampling data
initial_data = observer(search_space.sample(num_initial_points))






def create_bo_model(data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-7)
    return GaussianProcessRegression(gpr)


initial_models = trieste.utils.map_values(create_bo_model, initial_data)




pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim.threshold)
eci = trieste.acquisition.ExpectedConstrainedImprovement(
    OBJECTIVE, pof.using(CONSTRAINT)
    )


#### EGO TECHNIQUE #######
rule = EfficientGlobalOptimization(eci)  # type: ignore


num_steps = 20
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

data = bo.optimize(
    num_steps, initial_data, initial_models, rule, track_state=False
    ).try_get_final_datasets()




#final_model = data.try_get_final_models()[OBJECTIVE]


constraint_data = data[CONSTRAINT]
new_query_points = constraint_data.query_points[-num_steps:]
new_observations = constraint_data.observations[-num_steps:]
new_data = (new_query_points, new_observations)



#######################################





print("___________________________VISUALIZE 3D___________________________________")
from matplotlib import pyplot as plt

from util.inequality_constraints_utils import plot_init_query_points




plot_init_query_points(
    search_space,
    Sim,
    initial_data[OBJECTIVE].astuple(),
    initial_data[CONSTRAINT].astuple(),
    new_data,
)

plt.show()

plot_objective_and_constraints(search_space, Sim)
plt.show()

#test = final_model(1)