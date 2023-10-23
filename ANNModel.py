"""
This Module is created for managing the prediction of consumption of boilers.
The module uses an MLP Regressor model to infer the future consumption of the boiler in order to support optimization

Input of the services are csv files provided in the resource folder, aggregated together and properly adapted for our needs.
The output will constitute the prediction of usage in the following 24 hours(?)

To run the module

"""

import pickle

from utils import create_csv, filter_dataset

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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



