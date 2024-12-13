import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("AQI_Forecasting_ETS")

# Load and preprocess the dataset
df = pd.read_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/output/combined_data.csv', parse_dates=['timestamp'])

# Data preprocessing (same as before)
df['aqi_us'] = pd.to_numeric(df['aqi_us'], errors='coerce')
df = df.groupby('timestamp')['aqi_us'].mean().reset_index()
df.set_index('timestamp', inplace=True)
df = df.sort_index()
df = df.resample('H').mean().interpolate(method='linear')

def calculate_accuracy(y_true, y_pred):
    error = np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-5, None)
    accuracy = 100 - np.mean(error) * 100
    return accuracy

# Split the data
train_size = int(len(df) * 0.7)
train = df[:train_size]
test = df[train_size:]

def evaluate_ets_model(train_data, test_data, trend_type, seasonal_type, seasonal_periods=24):
    try:
        with mlflow.start_run(run_name=f"ETS_{trend_type}_{seasonal_type}"):
            # Log parameters
            mlflow.log_params({
                "trend_type": trend_type,
                "seasonal_type": seasonal_type,
                "seasonal_periods": seasonal_periods,
                "train_size": len(train_data),
                "test_size": len(test_data)
            })
            
            # Fit ETS model
            model = ExponentialSmoothing(
                train_data,
                trend=trend_type,
                seasonal=seasonal_type,
                seasonal_periods=seasonal_periods
            )
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mae = mean_absolute_error(test_data, forecast)
            accuracy = calculate_accuracy(test_data, forecast)
            
            # Log metrics
            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "accuracy": accuracy
            })
            
            # Create and save plot
            plt.figure(figsize=(12, 6))
            plt.plot(test_data.index, test_data, label='Actual AQI', color='blue')
            plt.plot(test_data.index, forecast, label='Forecasted AQI', color='red')
            plt.title('ETS Model Forecast vs Actual AQI')
            plt.xlabel('Time')
            plt.ylabel('AQI')
            plt.legend()
            plt.tight_layout()
            
            # Save plot and log as artifact
            plot_path = "C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/images/ets_forecast_plot.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            
            return forecast, rmse, mae, accuracy, fitted_model
            
    except Exception as e:
        print(f"Error in ETS model: {e}")
        return None, None, None, None, None

# Try different combinations
trend_types = ['add', 'mul', None]
seasonal_types = ['add', 'mul', None]

best_rmse = float('inf')
best_params = None
best_forecast = None
best_model = None
best_mae = None
best_accuracy = None

# Grid search with MLflow tracking
for trend in trend_types:
    for seasonal in seasonal_types:
        print(f"Testing ETS with trend={trend}, seasonal={seasonal}")
        
        forecast, rmse, mae, accuracy, model = evaluate_ets_model(
            train['aqi_us'],
            test['aqi_us'],
            trend,
            seasonal
        )
        
        if rmse is not None:
            print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (trend, seasonal)
                best_forecast = forecast
                best_model = model
                best_mae = mae
                best_accuracy = accuracy

# Log best model results
with mlflow.start_run(run_name="Best_ETS_Model"):
    if best_params is not None:
        mlflow.log_params({
            "best_trend": best_params[0],
            "best_seasonal": best_params[1]
        })
        mlflow.log_metrics({
            "best_rmse": best_rmse,
            "best_mae": best_mae,
            "best_accuracy": best_accuracy
        })
        print("\nBest ETS Model Parameters:")
        print(f"Trend: {best_params[0]}")
        print(f"Seasonal: {best_params[1]}")
        print(f"Best RMSE: {best_rmse:.2f}")
        print(f"Best MAE: {best_mae:.2f}")
        print(f"Best Accuracy: {best_accuracy:.2f}%")