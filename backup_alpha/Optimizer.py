import pickle
import time

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ANNModel import MLModel

from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.termination import Termination
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

"""
dataset = filter_dataset(r"s3_2_2_ENG/resources/InputDataframe.csv")
X = dataset.loc[0:,['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']]  #Le x e y della mia F
y = dataset['NG Consumption [kW]']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler().fit(X_train)

X_names_ann = ['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']
input_df = dataset.copy()

optimization_df = dataset.copy()  # ??

optimization_df = optimization_df[X_names_ann]
newnames = ['Q','Tb1','Tb2','Td','Hb1','Hb2']
optimization_df.columns = newnames
# optimization_df.head(5)
"""


class OptimizationProblem:

    def __init__(self, model, n_gen, pop_size):
        self.optimization_df = None
        self.scaler = None
        self.model = model
        self.X = 1
        self.n = len(self.optimization_df)
        self.fixed_value = 0.5
        self.n_gen = n_gen
        self.pop_size = pop_size
        
        self.lb_Tb1 = 0
        self.lb_Tb2 = 0
        self.lb_Td = 0
        self.lb_Hb1 = 0
        self.lb_Hb2 = 0

        self.ub_Tb1 = 85
        self.ub_Tb2 = 85
        self.ub_Td = 80
        self.ub_Hb1 = 0.5
        self.ub_Hb2 = 0.5

        self.size1 = 4500 * 0.8
        self.size2 = 4500 * 0.8

        self.lb_array = np.array([self.lb_Tb1, self.lb_Tb2, self.lb_Td, self.lb_Hb1, self.lb_Hb2] * self.n)
        self.ub_array = np.array([self.ub_Tb1, self.ub_Tb2, self.ub_Td, self.ub_Hb1, self.ub_Hb2] * self.n)

        self.termination = DefaultSingleObjectiveTermination(
            xtol = 1e-800,
            cvtol = 1e-600,
            ftol = 0.05,
            period = 200,
            n_max_gen = self.n_gen,
            n_max_evals = 1000000000
        )

        dataset = MLModel.filter_dataset(r"s3_2_2_ENG/resources/InputDataframe.csv")
        X = dataset.loc[0:,['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']]  #Le x e y della mia F
        y = dataset['NG Consumption [kW]']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        self.scaler = StandardScaler().fit(X_train)

        X_names_ann = ['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']
        input_df = dataset.copy()

        self.optimization_df = dataset.copy()  # ??

        self.optimization_df = self.optimization_df[X_names_ann]
        newnames = ['Q','Tb1','Tb2','Td','Hb1','Hb2']
        self.optimization_df.columns = newnames
        self.optimization_df = self.optimization_df[:self.n]

        # optimization_df.head(5)        

    def f(self, x):

        x_matrix = x.reshape((self.n, self.X))
        x_matrix = np.hstack((self.optimization_df['ENERGIA INSTANTANEA (15 minuto)'].values.reshape((n, 1)), x_matrix))
        x_matrix = pd.DataFrame(x_matrix, columns=['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERAS (15 minuto)'])

        # Apply the scaler transformation to the decision variables matrix
        x_matrix_scaled = self.scaler.transform(x_matrix)

        # Calculate the sum of the model predictions for all timesteps
        eta=model.predict(x_matrix_scaled)

        f=self.optimization_df['ENERGIA INSTANTANEA (15 minuto)'].values/eta

        return np.sum(f)
    
    def f_values(self, x):

        # Reshape the decision variables into a matrix with n rows and X columns
        x_matrix = x.reshape((self.n, self.X))
        
        x_matrix = np.hstack((self.optimization_df['ENERGIA INSTANTANEA (15 minuto)'].values.reshape((self.n, 1)), x_matrix))
        x_matrix = pd.DataFrame(x_matrix, columns=['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERAS (15 minuto)'])

    def g1(self, x):

        # Reshape the decision variables into a matrix with n rows and X columns
        x_matrix = x.reshape((self.n, self.X))
        
        # Calculate the constraint values for each timestep
        g = x_matrix[:, 2] - x_matrix[:, 1]

        g=np.max(g, axis=0)
        
        return g
    
    '''

    def g2(self, x):
        x_matrix = x.reshape((self.n, self.X))
        g = x_matrix[:, 2] - x_matrix[:, 0]
        g = np.max(g, axis=0)
        return g

    def g3(self, x):
        x_matrix = x.reshape((self.n, self.X))
        g = -(x_matrix[:, 3] * self.size1 + x_matrix[:, 4] * self.size2 - 0.5 * np.array(self.optimization_df['Q'].values))
        g = np.max(g, axis=0)
        return g

    '''

    def optimize(self):
        ngen = self.n_gen
        popsize = self.pop_size
        termination = self.termination
        best_objective_values = []
        algorithm = NSGA2(pop_size=self.pop_size)

        def callback(algorithm):
            print(f"Generation: {(100 * algorithm.n_gen / ngen):.2f}%")
            best_objective_value = algorithm.pop.get("F").min()
            best_objective_values.append(best_objective_value)

        problem = FunctionalProblem(self.X * self.n, self.f, constr_ieq=[self.g1, self.g2, self.g3], xl=self.lb_array, xu=self.ub_array)
        res = minimize(problem, algorithm, termination, seed=1, callback=callback)

        return res

if __name__ == "__main__":

    model = pickle.load(open(r"C:\Users\annatalini\OneDrive - Engineering Ingegneria Informatica S.p.A\DigiBUILD\DigiBUILD - Developement\s3_2_2_ENG\models\ANNTrainedModel.pkl", 'rb'))

    optimization_problem = OptimizationProblem(model, 100, 200)
    result = optimization_problem.optimize()

    print("Best solution found:", result.X)
    print("Objective value:", result.F)
    print("Constraint violation:", result.CV)
