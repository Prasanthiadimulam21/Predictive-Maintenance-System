from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Load models and scaler
rf_model = joblib.load('models/random_forest_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
stacking_model = joblib.load('models/stacking_model.pkl')  # Load the fitted stacking model
scaler = joblib.load('models/scaler.pkl')

# Feature names based on your dataset
feature_names = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

app = Flask(__name__)

# Input features function
def input_features():
    """Takes user input for features and ensures valid input."""
    feature_values = []
    for feature in feature_names:
        while True:
            try:
                value = float(request.form[feature])  # Get value from form
                feature_values.append(value)
                break
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a valid numerical value.")
    return feature_values

# Prediction function
def hybrid_predict_with_input():
    """Predict maintenance need using hybrid models based on user input."""
    input_data = input_features()
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_data_scaled = scaler.transform(input_df)

    # Predictions from both models
    rf_prediction = rf_model.predict(input_data_scaled)[0]
    svm_prediction = svm_model.predict(input_data_scaled)[0]

    # Weighted Averaging of Probabilities (70% RF, 30% SVM)
    rf_weight = 0.7
    svm_weight = 0.3
    rf_probs = rf_model.predict_proba(input_data_scaled)
    svm_probs = svm_model.predict_proba(input_data_scaled)
    weighted_probs = (rf_probs * rf_weight) + (svm_probs * svm_weight)
    weighted_prediction = np.argmax(weighted_probs, axis=1)[0]

    # Use the fitted stacking model for prediction
    stacking_prediction = stacking_model.predict(input_data_scaled)[0]

    # Results Interpretation
    rf_result = "Maintenance Needed" if rf_prediction == 1 else "No Maintenance Needed"
    svm_result = "Maintenance Needed" if svm_prediction == 1 else "No Maintenance Needed"
    weighted_result = "Maintenance Needed" if weighted_prediction == 1 else "No Maintenance Needed"
    stacking_result = "Maintenance Needed" if stacking_prediction == 1 else "No Maintenance Needed"

    return rf_result, svm_result, weighted_result, stacking_result

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        rf_result, svm_result, weighted_result, stacking_result = hybrid_predict_with_input()
        return render_template("result.html", 
                               rf_result=rf_result, 
                               svm_result=svm_result,
                               weighted_result=weighted_result,
                               stacking_result=stacking_result)
    return render_template("index.html", feature_names=feature_names)  # Pass feature names here

@app.route("/reports")
def reports():
    """Render the classification reports page."""
    # Load the reports from the static folder
    with open('static/random_forest_report.txt') as f:
        rf_report = f.read()
    
    with open('static/svm_report.txt') as f:
        svm_report = f.read()
    
    with open('static/stacking_model_report.txt') as f:
        stacking_report = f.read()
    
    return render_template("reports.html", 
                           rf_report=rf_report, 
                           svm_report=svm_report,
                           stacking_report=stacking_report)
    
@app.route("/visuals")
def visuals():
    """Render the visuals page."""
    return render_template("visuals.html")    

if __name__ == "__main__":
    app.run(debug=True)
