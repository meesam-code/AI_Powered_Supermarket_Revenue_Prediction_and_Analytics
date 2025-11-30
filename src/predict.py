import joblib
import pandas as pd

def predict(input_dict):
    model = joblib.load('models/model.pkl')
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]
