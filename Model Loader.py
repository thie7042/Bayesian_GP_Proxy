import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split


print('test')

filename = 'finalized_model.sav'

my_model = pickle.load(open(filename, 'rb'))

print(my_model)
print(type(my_model))

print('_________________________________')



query_point[0][0] = 0.2
query_point[0][1] = 0.4

result = bo.optimize(num_steps, query_point, my_model)
data = my_model.predict_y(1)