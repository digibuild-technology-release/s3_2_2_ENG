from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

#from Optimizer import run_optimization

# Creare un'istanza di Flask
app = Flask(__name__)
auth = HTTPBasicAuth()

# Caricare il modello di Machine Learning
model = pickle.load(open(r"s3_2_2_ENG\models\ANNTrainedModel.pkl", "rb"))

# Definire una funzione che viene utilizzata per verificare le credenziali dell'utente
@auth.verify_password
def verify_password(username, password):
    # Controllare se le credenziali sono corrette
    if username == "AndreaNatalini" and password == "Research.AN21_11":
        return True
    else:
        return False

@app.route("/test", methods=['GET'])
def test():
    return {"api":"connected"}

#metodo che restituisce l'input della rete
@app.route("/print_input", methods=['POST'])
@auth.login_required
def returnInputData():

    data = request.get_json()
    features = data['features'][1]
    return jsonify({"data": features})

# Creare una rotta nella Flask API
@app.route("/predict", methods=["POST"])
@auth.login_required
def predict():
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

'''
@app.route("/optimize", methods=["POST"])
@auth.login_required
def optimize():
    # Ottenere i dati in input dalla richiesta 
    data = request.get_json()

    # Eseguire la funzione di ottimizzazione
    solution = run_optimization(data)

    # Restituire la soluzione
    return jsonify(solution=solution)
'''

# Avviare la Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


