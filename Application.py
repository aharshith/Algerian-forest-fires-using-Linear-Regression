import pickle
import numpy as np
from flask import Flask, request, render_template

application = Flask(__name__)
app = application

# Load Ridge model and Scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Get data from form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale and predict
            new_data_scaled = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, classes, Region]])
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', results=result[0])
        except Exception as e:
            return f"Error occurred: {str(e)}"
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
