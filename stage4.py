import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

weather_data = pd.read_csv('usa_rain_prediction_dataset_2024_2025.csv')
X = weather_data.drop(columns=['Rain Tomorrow', 'Date', 'Location', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure'])
y = weather_data['Rain Tomorrow']


# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test dataset
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)

