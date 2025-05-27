import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Create models and static directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load dataset
data = pd.read_csv('sensor_data.csv')  # Adjust the path to your dataset location

# Data preprocessing: Handle missing values (if any) and scale features
if data.isnull().sum().any():
    data = data.fillna(method='ffill')

# Standardize features (except the target)
scaler = StandardScaler()
X = data.drop(['UDI', 'Product ID', 'Type', 'Machine failure'], axis=1)
X_scaled = scaler.fit_transform(X)
y = data['Machine failure']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)

# Initialize and train the SVM classifier with probability output
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Create the stacking model
base_models = [
    ('random_forest', rf_model),
    ('svm', svm_model)
]
meta_learner = LogisticRegression()
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_learner, cv=5)

# Fit the stacking model
stacking_model.fit(X_train, y_train)

# Save both models and the scaler for future use
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(stacking_model, 'models/stacking_model.pkl')  # Save the stacking model

print("Models and scaler saved as random_forest_model.pkl, svm_model.pkl, scaler.pkl, and stacking_model.pkl")

# Function to save confusion matrix
def save_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f'static/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()  # Close the plot to avoid display issues

# After evaluating models, save confusion matrices
save_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
save_confusion_matrix(svm_model, X_test, y_test, "SVM")
save_confusion_matrix(stacking_model, X_test, y_test, "Stacking Model")  # Add confusion matrix for stacking model

# Generate and save classification reports
def save_classification_report(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Failure", "Failure"])
    with open(f'static/{model_name.lower().replace(" ", "_")}_report.txt', 'w') as f:
        f.write(report)

# Save classification reports for Random Forest, SVM, and Stacking Model
save_classification_report(rf_model, X_test, y_test, "Random Forest")
save_classification_report(svm_model, X_test, y_test, "SVM")
save_classification_report(stacking_model, X_test, y_test, "Stacking Model")  # Save stacking report

# Feature Importance Plot
def save_feature_importance(model, feature_names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} Feature Importance")
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(f'static/{model_name.lower().replace(" ", "_")}_feature_importance.png')
    plt.close()

# Save feature importance for Random Forest
save_feature_importance(rf_model, X.columns, "Random Forest")
