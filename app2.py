from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load dynamic model
model = pickle.load(open("static/model.pkl", "rb"))

# Mapping transaction types to numeric values (must match training)
type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "OTHER": 5}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction_type = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])

        # Map transaction type to numeric
        val = type_mapping.get(transaction_type, 5)

        # Prepare input for prediction
        input_array = np.array([[val, amount, oldbalanceOrg, newbalanceOrig]])

        # Predict
        prediction = model.predict(input_array)
        output = int(prediction[0])

        # Human-readable result
        result = "Legitimate Transaction ✅" if output == 0 else "Fraudulent Transaction ⚠️"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
