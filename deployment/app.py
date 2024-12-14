from flask import Flask, request, render_template_string
import mlflow
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import compare_models

app = Flask(__name__)

# Load the best model and data
MODEL_TYPE, MODEL_PARAMS, BEST_RUN = compare_models()
print(f"Loaded {MODEL_TYPE} model with parameters: {MODEL_PARAMS}")

# Load historical data
df = pd.read_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/output/combined_data.csv', parse_dates=['timestamp'])
historical_data = pd.to_numeric(df['aqi_us'], errors='coerce')

# Define a simple HTML form
form_html = """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
            }
            form {
                margin: 20px 0;
            }
            input, button {
                padding: 8px;
                margin: 5px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .result {
                margin-top: 20px;
                padding: 10px;
                background-color: #fff;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>AQI Prediction</h2>
            <p>Current Model: {{ model_type }} (Accuracy: {{ accuracy }}%)</p>
            <form method="POST">
                <label for="days">How many days do you want to predict?</label><br>
                <input type="number" name="days" min="1" max="5" required><br>
                <button type="submit">Get Prediction</button>
            </form>
            {% if predictions %}
            <div class="result">
                <h3>Predictions for next {{ days }} day(s):</h3>
                <ul>
                {% for pred in predictions %}
                    <li>{{ pred['time'] }}: {{ "%.2f"|format(pred['value']) }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = None
    days = None
    
    if request.method == 'POST':
        days = int(request.form['days'])
        periods = days * 24  # Convert days to hours
        
        # Generate forecast based on model type
        if MODEL_TYPE == 'ARIMA':
            model = ARIMA(historical_data, order=(
                int(MODEL_PARAMS['p']),
                int(MODEL_PARAMS['d']),
                int(MODEL_PARAMS['q'])
            ))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=periods)
            
        else:  # ETS
            model = ExponentialSmoothing(
                historical_data,
                trend=MODEL_PARAMS['trend'],
                damped_trend=MODEL_PARAMS.get('damped_trend', 'False') == 'True'
            )
            fitted_model = model.fit(
                smoothing_level=float(MODEL_PARAMS.get('smoothing_level', 0.3)),
                smoothing_slope=float(MODEL_PARAMS.get('smoothing_slope', 0.1))
            )
            forecast = fitted_model.forecast(periods)
        
        # Create timestamps
        timestamps = pd.date_range(
            start=pd.Timestamp.now(),
            periods=len(forecast),
            freq='H'
        )
        
        # Format predictions
        predictions = [
            {'time': ts.strftime('%Y-%m-%d %H:%M'), 'value': val}
            for ts, val in zip(timestamps, forecast)
        ]
    
    return render_template_string(
        form_html,
        model_type=MODEL_TYPE,
        accuracy=round(BEST_RUN['metrics.accuracy'], 2),
        predictions=predictions,
        days=days
    )

if __name__ == '__main__':
    app.run(debug=True, port=8000)