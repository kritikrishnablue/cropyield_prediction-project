from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and transformer
model = joblib.load("crop_yield_model.pkl")
poly = joblib.load("poly_transformer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input year from the form
        year = int(request.form['year'])
        
        # Transform input using PolynomialFeatures
        year_poly = poly.transform(np.array([[year]]))

        # Make prediction
        prediction = model.predict(year_poly)

        return render_template('index.html', prediction=round(prediction[0], 2), year=year)

    except:
        return render_template('index.html', error="Invalid Input! Please enter a valid year.")

if __name__ == '__main__':
    app.run(debug=True)