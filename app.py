import logging
import pickle
import os
import glob
import threading
import sqlite3
import time

import numpy as np
import pandas as pd
from io import StringIO

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://"
)

UPLOAD_FOLDER = 'resources/requestData'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global scaler_fitted

class MLModel:

    """
    Class representing a machine learning model.

    Attributes:
        LHV (int): Lower Heating Value in kj/m3.
        LHV_ng (int): Lower Heating Value for natural gas.
        eta_lim (float): Limit for efficiency.
        zeros (int): Flag indicating whether to include zeros in the dataset.
        random_seed (int): Random seed for reproducibility.
        scaler (StandardScaler): Scaler object for data normalization.

    Methods:
        create_input(path: str, save_local_file: bool, **file_format: str) -> tuple:
            Create input data from files in the specified path and return processed dataframes.

        train_MLModel() -> tuple:
            Train the machine learning model and return the trained model, score, and scaler object.    
    """

    def __init__(self):

        self.LHV = 37411 # kj/m3
        self.LHV_ng = self.LHV # Used for conversion from m3/hr to 
        self.eta_lim = 1.3 
        self.zeros = 1
        self.random_seed = 1

        self.scaler = StandardScaler()

    # Creazione del dataframe


    def create_input(self, path: str, save_local_file: bool, **file_format:str): 

        """
        Create input data from files in the specified path and return processed dataframes.

        Args:
            path (str): The path to the files.
            save_local_file (bool): Flag indicating whether to save the processed dataset locally.
            **file_format (str): The file format to save the dataset in.

        Returns:
            tuple: A tuple containing three dataframes: df_30, dataset, and temp_df.
                - df_30: The processed dataframe with resampled data.
                - dataset: The filtered and transformed dataset.
                - temp_df: A temporary copy of the dataset before filtering.

        """

        #Genera una lista fatta da tutti i nomi che rientrano nella richiesta 

        nomifiles=(glob.glob(path))

        df=pd.DataFrame()

        for nomi in nomifiles:
            A0 = pd.read_csv(nomi, sep=';', header=None)
            df = pd.concat([df,A0])


        nomi_originali = df.iloc[:,2].unique() #Vediamo quante grandezze vengono studiate

        # elimina le colonne 'a' e 'b' dal dataframe
        df=df.drop(df.columns[0],axis=1)
        df=df.drop(df.columns[0],axis=1)


        df.columns = ['nome', 'orario', 'valore'] # Cambiamo il nome delle colonne

        df['valore'] = df['valore'].str.replace(',', '.') # Aggiustiamo i valori del dataframe 
        df['valore'] = df['valore'].str.strip() # Serve per togliere tutti gli spazi da quella colonna
        df['valore'] = df['valore'].astype(float) # Rendiamo la colonna dei numeri float


        # Crea un nuovo dataframe con gli orari come prima colonna

        df = df.pivot(index='orario', columns='nome', values='valore') #TODO questo pivot genera problemi

        df.index = pd.to_datetime(df.index)
        

        # Reimposta l'indice

        df_30 = df.resample('30T').mean()
        temp_1 = df_30.copy()
        # df_30=df.resample('15T').interpolate()


        #Limitati al temo di funzionamento B1-2

        df_30 = df_30.loc['2023-06-07 00:00:00':'2023-10-17 19:00:00']

        dataset = df_30.copy()

        dataset['NG Consumption [kW]'] = dataset['CONSUMO GAS (30 minutos)'].diff()*(self.LHV/1800)

        # dataset['NG Consumption [kW]'] = dataset['NG Consumption [kW]'].shift(-1)

        dataset['eta'] = dataset['ENERGIA INSTANTANEA (15 minuto)']/(dataset['NG Consumption [kW]']+0.001)
        dataset['Boiler 1 Hours'] = dataset['Horas Funcionamiento Caldera 1 (15 minuto)'].diff()
        dataset['Boiler 2 Hours'] = dataset['Horas Funcionamiento Caldera 2 (15 minuto)'].diff()
        dataset['Boiler 3 Hours'] = dataset['Horas Funcionamiento Caldera 3 (15 minuto)'].diff()
        dataset['Boiler 3 Hours'] = dataset['Boiler 3 Hours'].replace(np.nan, 0)

        dataset['BH'] = dataset['Boiler 1 Hours'] + dataset['Boiler 2 Hours'] #TODO ci serve davvero?

        dataset=pd.DataFrame(dataset)

        if self.zeros==1:
            #Elimino gli zeri da boiler hours
            dataset['filter'] = dataset.apply(lambda row: 0 if row['eta'] < self.eta_lim and
                                            row['ENERGIA INSTANTANEA (15 minuto)'] > 50
                                            and row['NG Consumption [kW]'] > 50
                                            #and row['BH'] > 0.05
                                            else 1, axis=1) #applica il se
        else:
            # Filtro ma lascio gli zeri
            dataset['filter'] = dataset.apply(lambda row: 0 if row['eta'] < self.eta_lim else 1, axis=1) # applica il se

        # Elimino i nan

        dataset = dataset.loc[dataset['filter'] != 1]
        dataset = dataset.drop('BH', axis=1)
        temp_df = dataset.copy()
        dataset = dataset.drop('filter', axis=1)

        dataset.fillna(0, inplace=True)

        if (save_local_file == True and file_format == '.xlsx'):
            dataset.to_excel('resources/TrainingDataset.xlsx')
        elif (save_local_file == True and file_format == '.csv'):
            dataset.to_csv('resources/TraningDataset.csv')
        else:
            pass

        return df_30, dataset, temp_df, temp_1
    
    def train_MLModel(self):

        """

        Train the machine learning model and return the trained model, score, and scaler object.

        Returns:
            tuple: A tuple containing the trained model, score, and scaler object.

        """

        dataset = self.create_input('resources/RVENA_23*.csv', save_local_file=False)[1]

        X = dataset.loc[:,['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERAS (15 minuto)']]  #Le x e y della mia F
        y = dataset.loc[:,['eta']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.random_seed)
        scaler_fitted = self.scaler.fit(X_train)
        X_train = scaler_fitted.transform(X_train)
        X_test = scaler_fitted.transform(X_test)

        # Definition of the ANN Model

        model = MLPRegressor(hidden_layer_sizes=(20, 100,300,100, 20),
                            max_iter=100000000,
                            verbose=True,
                            solver='adam',
                            learning_rate='adaptive',
                            random_state=self.random_seed,
                            activation='relu')
        
        trained_model = model.fit(X_train, y_train)

        pickle.dump(trained_model, open("resources/models/TrainedModel.pkl", 'wb'))

        # Valutazione delle prestazioni del modello sui dati di test

        score = trained_model.score(X_test, y_test)
        print(f'R^2 score: {score:.2f}')

        # Utilizzo del modello per fare previsioni sui dati di test

        X_pred = self.scaler.transform(X)    #Utilizzo lo stesso scaler che è stato fittato prima
        y_pred = model.predict(X_pred)

        return model, score, scaler_fitted

    # dataset = create_input('resources/RVENA_23*.csv', save_local_file=False)[1]
    # dataset.head()

class Optimizer:

    def __init__(self, dataset, model, n_gen, pop_size):
        
        self.optimization_df = dataset.iloc[:48].copy()
        self.X = 1
        self.n = len(dataset)
        self.start_o = 0
        self.final_df = self.optimization_df[self.start_o:self.start_o + self.n]
        self.model = model
        self.fixed_value = 0.5
        self.ngen = n_gen
        self.pop_size = pop_size
        self.random_seed = 1

        self.scaler = StandardScaler()
      
    def f(self, x):

        # Reshape the decision variables into a matrix with n rows and X columns
        x_matrix = x.reshape((self.n, self.X))

        X = self.optimization_df.loc[:,['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERAS (15 minuto)']]  #Le x e y della mia F
        y = self.optimization_df.loc[:,['eta']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.random_seed)
        scaler_fitted = self.scaler.fit(X_train)        
        
        x_matrix = np.hstack((self.final_df['ENERGIA INSTANTANEA (15 minuto)'].values.reshape((self.n, 1)), x_matrix))
        x_matrix = pd.DataFrame(x_matrix, columns=['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERAS (15 minuto)'])
       
        # Apply the scaler transformation to the decision variables matrix
        x_matrix_scaled = self.scaler.transform(x_matrix)
        
        # Calculate the sum of the model predictions for all timesteps
        eta=self.model.predict(x_matrix_scaled)
             
        f=self.final_df['ENERGIA INSTANTANEA (15 minuto)'].values/eta
        
        return np.sum(f)
    
    def f_values(self, x):

        # Reshape the decision variables into a matrix with n rows and X columns
        x_matrix = x.reshape((self.n, self.X))
        
        x_matrix = np.hstack((self.final_df['ENERGIA INSTANTANEA (15 minuto)'].values.reshape((self.n, 1)), x_matrix))
        x_matrix = pd.DataFrame(x_matrix, columns=['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERAS (15 minuto)'])

        
        # Apply the scaler transformation to the decision variables matrix
        x_matrix_scaled = self.scaler.transform(x_matrix)
        
        # Calculate the sum of the model predictions for all timesteps
        eta=self.model.predict(x_matrix_scaled)
        
        eta = np.clip(eta, a_min=None, a_max=1.3)

        f=self.final_df['ENERGIA INSTANTANEA (15 minuto)'].values/eta
    
        return f

    def g1(self, x):
        # Reshape the decision variables into a matrix with n rows and X columns
        x_matrix = x.reshape((self.n, self.X))
        
        # Calculate the constraint values for each timestep
        g = x_matrix[:, 2] - x_matrix[:, 1]
        g=np.max(g, axis=0)
        
        return g
    
    def optimize(self):
            
        self.termination = DefaultSingleObjectiveTermination(xtol=1e-800, cvtol=1e-600, ftol=0.05, period=200, n_max_gen=self.ngen, n_max_evals=1000000000)
        algorithm = NSGA2(pop_size=self.pop_size)
        self.best_objective_values = []  

        def callback(algorithm):

            print(f"Generation: {(100*algorithm.n_gen/self.ngen):.2f}%")
            best_objective_value = algorithm.pop.get("F").min()
            self.best_objective_values.append(best_objective_value)

        self.problem = FunctionalProblem(self.X * self.n, self.f, constr_ieq=[], xl=60, xu=90)

        start = time.time()
        logging.info("Starting Optimization")
        res = minimize(self.problem, algorithm, self.termination, seed=1, callback = callback)
        logging.info("Optimization Ended, Returning Results")
        end = time.time()
        
        total_time = (end-start)/60

        self.temperature = res.X.reshape(self.n,self.X)
        df_solutions = pd.DataFrame(self.temperature, columns=['Temperatures'])
        final_df = self.optimization_df[['ENERGIA INSTANTANEA (15 minuto)', 'VOLUMEN INSTANTANEO (15 minuto)', 'NG Consumption [kW]',  'eta', 'Boiler 1 Hours', 'Boiler 2 Hours','Boiler 3 Hours']]
        final_df['Optmized Temperatures'] = df_solutions['Temperatures'].values
        
        self.gas_real = final_df['NG Consumption [kW]'].sum()/2 # Total NG Consumption in kWh
        self.optimized_gas = res.F/2 # Optmized Gas Consumptiopn in kWh
        
        

        solution = {"Solution":{
            "realGas": f"Total NG Consumption {self.gas_real} kWh", 
            "OptimizedGas": f" Optmized Gas Consumption {int(self.optimized_gas)} kWh", 
            "Saved Gas": f"Total Gas Saved {float((self.gas_real-self.optimized_gas))} kWh",
            "Saved Cost": f"Saved Cost {float(100*(1-self.optimized_gas/self.gas_real))}%", #TODO specify value
            "Strategy": final_df.to_json(orient='columns'),
            "Total Execution Time": f"{(end-start)/60} seconds"
        }}
        
        return solution, df_solutions

@app.route('/status')
def check_status():
    return {'api':'connected'}

@app.route('/digibuild/s322/optimizer', methods=['POST'])
def upload_csv():

    try:
        # Verifica se la richiesta contiene file
        if 'csvFiles' not in request.files:
            return jsonify({'error': 'Nessun file nella richiesta'})

        files = request.files.getlist('csvFiles')

        # This block uploads csv files inside the directory
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        for file in files:
            if file and file.filename.endswith('.csv'):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                print('Caricamento avvenuto con successo')

        model = pickle.load(open('resources/models/TrainedModel.pkl', 'rb'))

        solution = Optimizer(MLModel().create_input('resources/requestData/RVENA_23*.csv', save_local_file=False)[1], model, 100, 200).optimize()[0]
        logging.info({'status': 'Ottimizzazione lanciata con successo'})

        # This block deletes all the files in the folder after the execution of the request
        files_in_folder = os.listdir(UPLOAD_FOLDER)
        for file_in_folder in files_in_folder:
            file_path = os.path.join(UPLOAD_FOLDER, file_in_folder)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Errore durante l'eliminazione del file {file_in_folder}: {e}")

        return solution
        
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/digibuild/s322/train-model', methods=['POST'])
def train_ml_model():

    try:

        # Verifica se la richiesta contiene un file CSV
        if 'csvFile' not in request.files:
            return jsonify({'error': 'Nessun file CSV nella richiesta'})

        file = request.files['csvFile']

        if file and file.filename.endswith('.csv'):
        
            dataset = MLModel().create_input('resources/requestData/RVENA_23*.csv', save_local_file=True, file_format='.csv')[1]
            r2_score = MLModel().train_MLModel()[1]

            response = {'status': {
                'R2 Score': r2_score
            }}

            return jsonify(response)
        
        else:
            return jsonify({'error': 'Il file deve essere in formato CSV'})
        
    except Exception as e:

        return jsonify({'error': str(e)})
    
@app.route('/digibuild/s322/optimizer_test', methods = ['POST'])
def optimizer_test():

    try:

        # Verifica se la richiesta contiene un file CSV
        if 'csvFile' not in request.files:
            
            start = time.time()

            model = pickle.load(open('resources/models/TrainedModel.pkl', 'rb'))
            dataset = MLModel().create_input(r'resources/requestData/RVENA_23*.csv', save_local_file=True)[2].iloc[:48]
            optimizer = Optimizer(dataset, model, 100, 200).optimize()
            solution = optimizer[0]
            df_temps = optimizer[1]
            # dataset['Optmized Temperatures'] = df_temps['Temperature']
            
            end = time.time()
            
            total_time = (end-start)/60

            return jsonify(solution)

        file = request.files['csvFile']

        if file and file.filename.endswith('.csv'):

            model = pickle.load(open('resources/models/TrainedModel.pkl', 'rb'))

            solution = Optimizer(MLModel().create_input('resources/requestData/RVENA_23*.csv', save_local_file=False)[3].iloc[:48], model, 100, 200).optimize()[0]
            logging.info({'status': 'Ottimizzazione lanciata con successo'})

            return jsonify(solution)
               
        else:
            return jsonify({'error': 'Il file deve essere in formato CSV'})
        
    except Exception as e:

        return jsonify({'error': str(e)})

if __name__ == '__main__':

    # dataset = MLModel().create_input('resources/dataset/RVENA_23*.csv', save_local_file=False)[1]
    # print(dataset.head(10))
    # model = pickle.load(open('resources/models/TrainedModel.pkl', 'rb'))
    # r2_score = MLModel().train_MLModel()[1]
    # print(r2_score)

    # solution = Optimizer(dataset, model, 100, 200).optimize()[0]
    # print(solution)

    app.run(host="0.0.0.0", port = 5000, debug = True)