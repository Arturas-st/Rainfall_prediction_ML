import pandas as pd
import joblib

#Loading persistent model
model = joblib.load('rain_prediction.joblib')

#creates input data
input_data = pd.DataFrame([[36, 72]], columns=['Temperature', 'Humidity'])

#making predictions
prediction = model.predict(input_data)

print(f"Predicted rain: {prediction[0]}")
