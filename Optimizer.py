
import pickle
import pandas as pd
import numpy as np
from utils import filterDataset

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem


from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization import scatter

input_df = filterDataset("resources/input_df.csv")

X_names_ann=['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']

n_times=30

df_opt=input_df[X_names_ann]
df_opt=df_opt.iloc[:n_times]
X_names_opt=['Q','Tb1','Tb2','Td','Hb1','Hb2']

model = pickle.load(open("resources\TrainedModel.pkl", 'rb'))

# NSGA-II

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=6,
                         n_obj=1,
                         n_ieq_constr=2,
                         xl=np.array([0, 0, 0, 0, 0, 0]),
                         xu=np.array([1000, 100, 100, 100, 0.5, 0.5]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = model.predict(np.array([x[0], x[1], x[2], x[3], x[4], x[5]]).reshape(1, -1))

        g1 = x[3] - x[2]
        g2 = x[3] - x[1]

        out["F"] = [f1]
        out["G"] = [g1, g2]   

problem = MyProblem()
algorithm = NSGA2(pop_size = 50)

from pymoo.core.callback import Callback
from pymoo.termination.default import DefaultMultiObjectiveTermination

# definisci un callback per salvare i valori della funzione obiettivo ad ogni generazione
class SaveObjectivesCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.objectives = []

    def notify(self, algorithm):
        self.objectives.append(algorithm.pop.get("F"))

# crea un'istanza del callback
save_objectives_callback = SaveObjectivesCallback()

# esegui l'ottimizzazione
res = minimize(problem,
               algorithm,
               seed=1,
               callback=save_objectives_callback,
               save_history=True,
               verbose=True)

# visualizza i valori della funzione obiettivo in un grafico
import matplotlib.pyplot as plt

plt.plot(save_objectives_callback.objectives)
plt.xlabel('Generation')
plt.ylabel('Objective Values')
plt.show()

