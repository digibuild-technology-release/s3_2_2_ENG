import pandas as pd
import glob 
import pickle

from utils import createCsv, filterDataset

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


createCsv(r's3_2_2_ENG\resources\RVENA*.csv')
input_df = filterDataset(r"s3_2_2_ENG\resources\input_df.csv")

X = input_df.loc[0:,['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']]  #Le x e y della mia F
y = input_df['NG Consumption [kW]']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# DataScaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model Initialization
model = MLPRegressor(hidden_layer_sizes=(100, 200, 100), max_iter=1000000,activation='relu')
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f'R^2 score: {score:.2f}')

X_pred = scaler.transform(X)
y_pred = model.predict(X_pred)

# Save Model
filename = r's3_2_2_ENG\models\ANNTrainedModel.pkl'
pickle.dump(model, open(filename, 'wb'))

print(y_pred)



