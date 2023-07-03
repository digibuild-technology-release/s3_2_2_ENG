import pickle

import pandas as pd
import numpy as np


from flask import Flask, jsonify, request
from flask_api import status
from models import db, CallLog
from VeoliaSERVICE import MLPModel


app = Flask(__name__)
app.config.from_pyfile('config.py')

input_data = pd.read_csv("resources/input_df.csv")
input_data = MLPModel().filterDataset()

X = np.array(input_data[['ENERGIA INSTANTANEA (15 minuto)',
                         'TEMP IMP CALDERA 1 (15 minuto)',
                         'TEMP IMP CALDERA 2 (15 minuto)',
                         'TEMPERATURA IMPULSION ANILLO (15 minuto)',
                         'Boiler 1 Hours','Boiler 2 Hours']])#Le x e y della mia F

y = np.array(input_data['NG Consumption [kW]'])

db.init_app(app)

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/calls', methods = ['GET'])
def getCalls():
    # Recupera le chiamate registrate dal database utilizzando il modello CallLog
    calls = CallLog.query.all()

    # Elabora i dati delle chiamate registrate come desiderato
    data = []
    for call in calls:
        call_data = {
            'method': call.method,
            'url': call.url,
            'parameters': call.parameters,
            'body': call.body,
            'timestamp': call.timestamp
        }
        data.append(call_data)

    # Restituisci i dati delle chiamate come risposta alla richiesta GET
    return jsonify(data)    


@app.route('/veolia/neural_network/metrics', methods=['GET'])
def getMetrics():

    model = pickle.load(open("resources\TrainedModel.pkl", 'rb'))
    pred = MLPModel().predict(model, X)

    metrics = MLPModel().get_metrics(trainedModel=model, prediction=pred, y_test=y)

    return metrics

@app.route('/veolia/riovena/optimization', methods=['GET'])
def getOptimizationStrategy():

    return jsonify({"Charging Strategy":{
    "Vehicle To Recharge" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Column to Use" : [1, 2],
    "Percentage of PV usage": 50 
    }})

@app.route('/veolia/riovena/train_model', methods =['GET'])
def trainModel():

    try:
        X_scaled, y_scaled = MLPModel().scale_data(X, y)
        trainedModel = MLPModel(X_scaled, y_scaled).train(testSize=0.3, randomState=42)
        return {"Model" : "trained"}
    
    except Exception as e:
        
        return {"Exception": e}
    
@app.route("/veolia/riovena/train_model", methods =['GET'])
def makePrediction():

    prediction = MLPModel().predict(X)
    return prediction

if __name__ == '__main__':
    app.run()