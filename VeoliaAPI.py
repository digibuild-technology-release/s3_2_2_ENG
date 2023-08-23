from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Set up Flask
app = Flask(__name__)
auth = HTTPBasicAuth()

# Load ML Model
model = pickle.load(open(r"s3_2_2_ENG\models\ANNTrainedModel.pkl", "rb"))

# Load Optimizer
problem = pickle.load(open(r"s3_2_2_ENG\models\OptimizerProblemTest.pkl", "rb"))
algorithm = pickle.load(open(r"s3_2_2_ENG\models\OptimizerAlgorithmTest.pkl", "rb"))
termination = pickle.load(open(r"s3_2_2_ENG\models\OptimizerTerminationTest.pkl", "rb"))

# Verify Credentials
@auth.verify_password
def verify_password(username, password):
    # Controllare se le credenziali sono corrette
    if username == "AndreaNatalini" and password == "Research.AN21_11":
        return True
    else:
        return False

# Test Endpoint
@app.route("/test", methods=['GET'])
def test():
    return {"api":"connected"}

# Display Input Data
@app.route("/print_input", methods=['POST'])
@auth.login_required
def returnInputData():

    data = request.get_json()
    features = data['features'][1]
    return jsonify({"data": features})

# Prediction Endpoint
@app.route("/predict", methods=["POST"])
@auth.login_required
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
            return jsonify({"prediction":list(prediction)})
        except ValueError as e:
            return jsonify(error=str(e))
        except Exception as e:
            return jsonify(error=str(e))
    else:
        return {"features shape": features.shape}


# Optimization Endpoint
@app.route("/optimize", methods=["POST"])
@auth.login_required
def optimize():

    import time
    from pymoo.optimize import minimize 
    # Ottenere i dati in input dalla richiesta 
    data = request.get_json()

    # Eseguire la funzione di ottimizzazione
    start_time = time.time()
    res = minimize(problem, algorithm, termination, seed=1, callback=callback)
    end_time = time.time()

    totalTime = end_time-start_time*60
    
    # Restituire la soluzione
    return jsonify({"solution":list(res.X),
                    "exxectuion time": totalTime})

# Avviare la Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


