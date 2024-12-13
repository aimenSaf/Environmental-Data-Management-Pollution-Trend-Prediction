import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri("file:./mlruns")
# Set the experiment name
mlflow.set_experiment("AQI_Forecasting_ARIMA")

# Load and preprocess the dataset
df = pd.read_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/output/combined_data.csv', parse_dates=['timestamp'])

# Convert aqi_us to numeric and handle duplicates
df['aqi_us'] = pd.to_numeric(df['aqi_us'], errors='coerce')
df = df.groupby('timestamp')['aqi_us'].mean().reset_index()
df.set_index('timestamp', inplace=True)
df = df.sort_index()

# Resample to hourly frequency
df = df.resample('H').mean().interpolate(method='linear')
df = df.dropna()

def calculate_accuracy(y_true, y_pred):
    error = np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-5, None)
    accuracy = 100 - np.mean(error) * 100
    return accuracy

def arima_model(data, p, d, q):
    train_size = int(len(data) * 0.7)
    train, test = data[:train_size], data[train_size:]
    
    if len(train) < 4:
        return None, None, None, None, None, None
    
    try:
        with mlflow.start_run(run_name=f"ARIMA_{p}_{d}_{q}"):
            # Log parameters
            mlflow.log_params({
                "p": p,
                "d": d,
                "q": q,
                "train_size": train_size,
                "total_samples": len(data)
            })
            
            # Fit the model
            model = ARIMA(train, order=(p, d, q))
            model_fit = model.fit()
            
            # Make predictions
            forecast = model_fit.forecast(steps=len(test))
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mae = mean_absolute_error(test, forecast)
            accuracy = calculate_accuracy(test, forecast)
            
            # Log metrics
            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "accuracy": accuracy
            })
            
            # Create and save plots
            plt.figure(figsize=(10, 6))
            plt.plot(test.index, test, label='Actual AQI', color='blue')
            plt.plot(test.index, forecast, label='Forecasted AQI', color='red')
            plt.title('ARIMA Model Forecast vs Actual AQI')
            plt.xlabel('Time')
            plt.ylabel('AQI')
            plt.legend()
            plt.tight_layout()
            
            # Save plot and log as artifact
            plot_path = "C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/images/arima_forecast_plot.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            
            return train, test, forecast, rmse, mae, accuracy
            
    except Exception as e:
        print(f"Error in ARIMA model: {e}")
        return None, None, None, None, None, None

# Model training and evaluation
best_rmse = float('inf')
best_params = None
best_forecast = None
best_train = None
best_test = None
best_mae = None
best_accuracy = None

# Grid search with MLflow tracking
for p in range(0, 2):
    for d in range(0, 2):
        for q in range(0, 2):
            train, test, forecast, rmse, mae, accuracy = arima_model(df['aqi_us'], p, d, q)
            
            if rmse is not None:
                print(f"ARIMA({p},{d},{q}) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (p, d, q)
                    best_forecast = forecast
                    best_train = train
                    best_test = test
                    best_mae = mae
                    best_accuracy = accuracy

# Log best model results
with mlflow.start_run(run_name="Best_ARIMA_Model"):
    if best_params is not None:
        mlflow.log_params({
            "best_p": best_params[0],
            "best_d": best_params[1],
            "best_q": best_params[2]
        })
        mlflow.log_metrics({
            "best_rmse": best_rmse,
            "best_mae": best_mae,
            "best_accuracy": best_accuracy
        })
        print("\nBest ARIMA Model:", best_params)
        print(f"Best RMSE: {best_rmse:.2f}")
        print(f"Best MAE: {best_mae:.2f}")
        print(f"Best Accuracy: {best_accuracy:.2f}%")