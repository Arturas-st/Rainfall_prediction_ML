import pandas as pd

#Import data
weather_data = pd.read_csv('usa_rain_prediction_dataset_2024_2025.csv')

#Identifies missing values
print(weather_data.info()) 
print(weather_data.isnull().sum()) 

