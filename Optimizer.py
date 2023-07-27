#%%Importing Libraries
import pickle
import time

import pandas as pd
import numpy as np

from utils import filterDataset, createCsv
from ANNModel import fitted_scaler

from sklearn.preprocessing import StandardScaler

from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

#%%Create Necessary CSV
createCsv(r's3_2_2_ENG\resources\RVENA_23*.csv')
dataset = filterDataset(r"s3_2_2_ENG\resources\InputDataframe.csv")
model = pickle.load(open(r"s3_2_2_ENG\models\ANNTrainedModel.pkl", 'rb'))

X_names_ann = ['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']
input_df = dataset.copy()

optimization_df = dataset.copy() #??

optimization_df = optimization_df[X_names_ann]
newnames = ['Q','Tb1','Tb2','Td','Hb1','Hb2']
optimization_df.columns = newnames

#%% Set Decisional Variables
X=5 #Set how much "kinds" of decisional variables are
n = 372 # Set the number of "timestep" for each decisional variable
fixed_value = 0.5 # Set the fixed value for the first constraint

# Define the lower bounds for each decision variable
lb_Tb1 = 0
lb_Tb2 = 0
lb_Td = 0
lb_Hb1 = 0
lb_Hb2 = 0

# Define the upper bounds for each decision variable
ub_Tb1 = 85
ub_Tb2 = 85
ub_Td = 80
ub_Hb1 = 0.5
ub_Hb2 = 0.5

size1=4500*0.8   #4500kW
size2=4500*0.8    #4500 kW   *3600*MW -> MJ

# Create the lower and upper bound arrays
lb_array = np.array([lb_Tb1, lb_Tb2, lb_Td, lb_Hb1, lb_Hb2] * n)
ub_array = np.array([ub_Tb1, ub_Tb2, ub_Td, ub_Hb1, ub_Hb2] * n)

#%% Define Functions
def f(x):
    # Reshape the decision variables into a matrix with n rows and X columns
    x_matrix = x.reshape((n, X))

    # Add the known Q variable as the first column of the matrix
    x_matrix = np.hstack((optimization_df['Q'].values.reshape((n, 1)), x_matrix))

    # Apply the scaler transformation to the decision variables matrix
    x_matrix_scaled = fitted_scaler.transform(x_matrix)

    # Calculate the sum of the model predictions for all timesteps
    return np.sum(model.predict(x_matrix_scaled))

g1 = lambda x: np.sum(x.reshape((n, X))[:, 2] - x.reshape((n, X))[:, 1])
g2 = lambda x: np.sum(x.reshape((n, X))[:,2] - x.reshape((n, X))[:, 0])

def g3(x, size1=4500*0.8, size2=4500*0.8):
    # Reshape the decision variables into a matrix with n rows and X columns
    x_matrix = x.reshape((n, X))

    # Calculate the constraint values for each timestep
    g = np.sum(-(x_matrix[:, 3] * size1 + x_matrix[:, 4] * size2 - 0.5 * np.array(optimization_df['Q'].values)))

    return g


#%% Optimization problem definition

ngen=150
popsize=100

termination = DefaultSingleObjectiveTermination(
    xtol=1e-800,
    cvtol=1e-600,
    ftol=0.05,
    period=200,
    n_max_gen=ngen,
    n_max_evals=1000000000
)

best_objective_values = []

algorithm = NSGA2(pop_size=popsize)

def callback(algorithm):
    print(f"Generation: {(100*algorithm.n_gen/ngen):.2f}%")
    best_objective_value = algorithm.pop.get("F").min()
    best_objective_values.append(best_objective_value)

#%% Optimization Iteration

start_time = time.time()

# Run the optimization with a maximum of 50 generations
# Limit the dataframe
optimization_df = optimization_df[:n]
problem = FunctionalProblem(X * n, f, constr_ieq=[g1,g2,g3], xl=lb_array, xu=ub_array)
res = minimize(problem, algorithm, termination, seed=1, callback=callback)

#%% Print Results

# Print the results
print("Best solution found:", res.X)
print("Objective value:", res.F)
print("Constraint violation:", res.CV)