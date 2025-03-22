import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

weather_data = pd.read_csv('usa_rain_prediction_dataset_2024_2025.csv')
X = weather_data.drop(columns=['Rain Tomorrow', 'Date', 'Location', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure'])
y = weather_data['Rain Tomorrow']

#Create the model
model = DecisionTreeClassifier()
model.fit(X,y)

#Create trained/persistent model 
joblib.dump(model, 'rain_prediction.joblib')

