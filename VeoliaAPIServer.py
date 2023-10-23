import logging
import pickle
import json

import numpy as np

from flask import Flask, request, jsonify
from flask_oidc import OpenIDConnect

from keycloak import KeycloakOpenID

from utils import filter_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.DEBUG)

# Set up Flask
app = Flask(__name__)
app.config.update({
    'SECRET_KEY': 'vqalcXI09J0EAfNdpqUFnwaFtihDNnrH',
    'TESTING': True,
    'DEBUG': True,
    'OIDC_CLIENT_SECRETS': 'auth.json',
    'OIDC_ID_TOKEN_COOKIE_SECURE': False,
    'OIDC_REQUIRE_VERIFIED_EMAIL': False,
    'OIDC_USER_INFO_ENABLED': True,
    'OIDC_OPENID_REALM': 'flask-app',
    'OIDC_SCOPES': ['openid', 'email', 'profile'],
    'OIDC_TOKEN_TYPE_HINT': 'access_token',
    'OIDC_INTROSPECTION_AUTH_METHOD': 'client_secret_post'
    # 'OIDC_INTROSPECTION_AUTH_METHOD': 'bearer'
})

oidc = OpenIDConnect(app)

keycloak_openid = KeycloakOpenID(server_url="http://localhost:8080/",
                                 client_id="Flask",
                                 realm_name="Flask-app",
                                 client_secret_key="vqalcXI09J0EAfNdpqUFnwaFtihDNnrH")

# Load ML Model
model = pickle.load(open(r"models/ANNTrainedModel.pkl", "rb"))

# Load Dataframes
optimization_df = filter_dataset(r"resources/InputDataframe.csv")
X = optimization_df.loc[0:, ['ENERGIA INSTANTANEA (15 minuto)', 'TEMP IMP CALDERA 1 (15 minuto)',
                             'TEMP IMP CALDERA 2 (15 minuto)', 'TEMPERATURA IMPULSION ANILLO (15 minuto)',
                             'Boiler 1 Hours', 'Boiler 2 Hours']]  # Le x e y della mia F

y = optimization_df['NG Consumption [kW]']

X_names_ann = ['ENERGIA INSTANTANEA (15 minuto)',
               'TEMP IMP CALDERA 1 (15 minuto)',
               'TEMP IMP CALDERA 2 (15 minuto)',
               'TEMPERATURA IMPULSION ANILLO (15 minuto)',
               'Boiler 1 Hours',
               'Boiler 2 Hours']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler().fit(X_train.values)

optimization_df = optimization_df[X_names_ann]
new_names = ['Q', 'Tb1', 'Tb2', 'Td', 'Hb1', 'Hb2']
optimization_df.columns = new_names

global prediction


# Test Endpoint
@app.route("/test", methods=['GET'])
@oidc.require_login
def test():
    return {"api": "connected"}


# Display Input Data
@app.route("/print_input", methods=['POST'])
def return_input_data():

    data = request.get_json()
    features = data['features'][1]
    return jsonify({"data": features})


# Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():

    # Creare una variabile globale che potrà essere utilizzata in seguito
    global prediction

    # Ottenere i dati in input dalla richiesta
    data = request.get_json()
    features = np.array(data["features"])

    if features.shape[1] == 6:
        # Stimare il valore del dato
        try:
            prediction = model.predict(features)
            # Restituire la predizione
            return jsonify({"prediction": list(prediction)})
        except ValueError as e:
            return jsonify(error=str(e))
        except Exception as e:
            return jsonify(error=str(e))
    else:
        return {"features shape": features.shape}


# Optimization Endpoint
@app.route("/optimize", methods=["POST"])
def optimize():

    from Optimizer import OptimizationProblem
    import time

    # Ottenere i dati in input dalla richiesta 
    data = request.get_json()

    # Eseguire la funzione di ottimizzazione

    start_time = time.time()
    
    optimization_problem = OptimizationProblem(optimization_df, scaler, model)
    result = optimization_problem.optimize()

    end_time = time.time()

    total_time = (end_time-start_time)*60
    
    # Restituire la soluzione
    return jsonify({"Best Solution": list(result.X),
                    "Execution Time": total_time})


# Avviare la Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
