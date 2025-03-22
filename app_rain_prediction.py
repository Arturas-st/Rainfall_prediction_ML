from flask import Flask, request, render_template
import joblib

model = joblib.load('rain_prediction.joblib')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the request
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    # Preprocess the input features
    # Make predictions using the loaded model
    prediction = model.predict([[temp, humidity]])
    result = "Rainfall" if prediction[0] == 1 else "No Rainfall"
    return render_template('index.html', prediction_result=result)


if __name__ == '__main__':
    app.run(debug=True)
