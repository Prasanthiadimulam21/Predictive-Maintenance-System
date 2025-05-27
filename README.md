# Predictive Maintenance System

A web-based predictive maintenance platform designed to analyze IoT sensor data from industrial machines and predict equipment failures before they occur. This project leverages machine learning algorithms and a Flask-based web interface to provide real-time failure predictions.

## 🔧 Features

- Upload and analyze industrial IoT sensor data
- Predict potential machine failures with over 95% accuracy
- Visualize results and insights in a clean web interface
- Built-in data preprocessing and feature selection
- RESTful API endpoints for integration
- Simple and intuitive UI built with HTML/CSS and Flask

## 🚀 Technologies Used

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Flask**
- **Matplotlib / Seaborn** (for optional data visualization)

## 🧠 Machine Learning Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (optional)

## 📁 Project Structure

```
predictive-maintenance/
│
├── app.py                # Flask backend
├── templates/
│   └── index.html        # Frontend template
├── static/
│   └── style.css         # Stylesheet
├── models/
│   └── model.pkl         # Trained model file
├── data/
│   └── sample_data.csv   # Example sensor data
└── README.md             # This file
```

## ⚙️ How to Run

1. Clone the repository:
   ```
   git clone https://github.com/your-username/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```
   python app.py
   ```

4. Open your browser and go to `http://127.0.0.1:5000`

## 📊 Sample Output

- Failure prediction: ✔️ / ❌
- Accuracy: 95%
- Confusion matrix and classification report shown after training

## 📌 Future Improvements

- Integration with real-time sensor data streams
- Admin dashboard for monitoring machine health
- Cloud deployment using Docker or Heroku



## 📬 Contact

For queries or suggestions, feel free to contact [patanthaseen2004@gmail.com].
