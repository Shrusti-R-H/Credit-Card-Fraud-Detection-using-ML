import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_fraud(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_array[:, [0, -1]] = scaler.transform(input_array[:, [0, -1]])
    prediction = model.predict(input_array)
    return prediction[0]
