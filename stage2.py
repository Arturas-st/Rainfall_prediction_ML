import pandas as pd

weather_data = pd.read_csv('usa_rain_prediction_dataset_2024_2025.csv')

#remove unwanted columns
X = weather_data.drop(columns=['Rain Tomorrow', 'Date', 'Location', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure'])

#labels
y = weather_data['Rain Tomorrow']

print(y)



