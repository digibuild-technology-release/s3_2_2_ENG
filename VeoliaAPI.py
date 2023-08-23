from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from utils import filterDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Set up Flask
app = Flask(__name__)
auth = HTTPBasicAuth()

# Load ML Model
model = pickle.load(open(r"s3_2_2_ENG\models\ANNTrainedModel.pkl", "rb"))

# Load Dataframes
optimization_df = filterDataset(r"C:\Users\annatalini\Documents\DigiBUILD-Developement\s3_2_2_ENG\resources\InputDataframe.csv")
X = optimization_df.loc[0:,['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']]  #Le x e y della mia F
y = optimization_df['NG Consumption [kW]']
X_names_ann = ['ENERGIA INSTANTANEA (15 minuto)','TEMP IMP CALDERA 1 (15 minuto)','TEMP IMP CALDERA 2 (15 minuto)','TEMPERATURA IMPULSION ANILLO (15 minuto)','Boiler 1 Hours','Boiler 2 Hours']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler().fit(X_train.values)

optimization_df = optimization_df[X_names_ann]
newnames = ['Q','Tb1','Tb2','Td','Hb1','Hb2']
optimization_df.columns = newnames

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

    from Optimizer import OptimizationProblem
    import time

    # Ottenere i dati in input dalla richiesta 
    data = request.get_json()

    # Eseguire la funzione di ottimizzazione

    start_time = time.time()
    
    optimization_problem = OptimizationProblem(optimization_df, scaler, model)
    result = optimization_problem.optimize()

    end_time = time.time()

    totalTime = (end_time-start_time)*60
    
    # Restituire la soluzione
    return jsonify({"Best Solution":list(result.X),
                    "Execution Time": totalTime})

# Avviare la Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


