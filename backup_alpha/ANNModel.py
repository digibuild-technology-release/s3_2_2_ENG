"""
This Module is created for managing the prediction of consumption of boilers.
The module uses an MLP Regressor model to infer the future consumption of the boiler in order to support optimization

Input of the services are csv files provided in the resource folder, aggregated together and properly adapted for our needs.
The output will constitute the prediction of usage in the following 24 hours(?)

To run the module

"""

import pickle
import logging
import glob

import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MLModel:

    def __init__():

        pass


    def create_csv(input_folder):

        """
        This method creates an input csv considering all the available data in teh folder resources, cleaned,
        and filtered for 30 minutes slices.
        The final result is called input_df

        input_folder = 'resources/RVENA_23*.csv'
        """

        nomifiles = (glob.glob(input_folder))
        df = pd.DataFrame()

        for nomi in nomifiles:
            a0 = pd.read_csv(nomi, sep=';', header=None)
            df = pd.concat([df, a0])

        nomi_originali = df.iloc[:, 2].unique()  # Vediamo quante grandezze vengono studiate
        # elimina le colonne 'a' e 'b' dal dataframe
        df = df.drop(df.columns[0], axis=1)
        df = df.drop(df.columns[0], axis=1)
        df.columns = ['nome', 'orario', 'valore']  # Cambiamo il nome delle colonne
        df['valore'] = df['valore'].str.replace(',', '.')  # Aggiustiamo i valori del dataframe
        df['valore'] = df['valore'].str.strip()  # Serve per togliere tutti gli spazi da quella colonna
        df['valore'] = df['valore'].astype(float)  # Rendiamo la colonna dei numeri float
        # Crea un nuovo dataframe con gli orari come prima colonna
        df = df.pivot(index='orario', columns='nome', values='valore')
        df.index = pd.to_datetime(df.index)
        # Reimposta l'indice
        df_30 = df.resample('30T').mean()
        # Salva il dataframe
        df_30.to_csv(r"resources/InputDataframe.csv")

    def filter_dataset(input_df):

        """
        input_df = "resources/input_df.csv"
        """

        lvh_ng = 9.5  # used for conversion from m3/hr to
        eta_lim = 1.40

        dataset = pd.read_csv(input_df)
        dataset['NG Consumption [kW]'] = dataset['CONSUMO GAS (30 minutos)'].diff() * lvh_ng*2
        dataset['eta'] = dataset['ENERGIA INSTANTANEA (15 minuto)']/(dataset['NG Consumption [kW]']+1)
        dataset['Boiler 1 Hours'] = dataset['Horas Funcionamiento Caldera 1 (15 minuto)'].diff()
        dataset['Boiler 2 Hours'] = dataset['Horas Funcionamiento Caldera 2 (15 minuto)'].diff()
        dataset['BH'] = dataset['Boiler 1 Hours']+dataset['Boiler 2 Hours']

        # dataset['filter'] = dataset.apply(lambda row: 0 if row['eta'] < eta_lim and row['BH'] > 0 else 1, axis=1)
        # #applica il se

        dataset['filter'] = dataset.apply(lambda row: 0 if row['eta'] < eta_lim else 1, axis=1)  # applica il se

        dataset = dataset.loc[dataset['filter'] != 1]

        dataset = dataset.drop('BH', axis=1)
        dataset = dataset.drop('filter', axis=1)

        dataset.fillna(0, inplace=True)

        return dataset    


    def train_model(self):

        create_csv(r'resources/RVENA*.csv')
        input_df = filter_dataset(r"resources/input_df.csv")

        X = input_df.loc[0:,
        ['ENERGIA INSTANTANEA (15 minuto)',
            'TEMP IMP CALDERA 1 (15 minuto)',
            'TEMP IMP CALDERA 2 (15 minuto)',
            'TEMPERATURA IMPULSION ANILLO (15 minuto)',
            'Boiler 1 Hours',
            'Boiler 2 Hours']]

        y = input_df['NG Consumption [kW]']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # DataScaling using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Initialization
        model = MLPRegressor(hidden_layer_sizes=(100, 200, 100), max_iter=1000000, activation='relu')
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f'R^2 score: {score:.2f}')

        X_pred = scaler.transform(X)
        y_pred = model.predict(X_pred)

        # Save Model
        filename = r'models/ANNTrainedModel_Test.pkl'
        pickle.dump(model, open(filename, 'wb'))

        print(y_pred)