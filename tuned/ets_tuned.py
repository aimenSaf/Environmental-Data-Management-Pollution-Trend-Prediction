# ets_tuning.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Set MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("AQI_Forecasting_ETS_Tuning")

# Load and preprocess data
df = pd.read_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/output/combined_data.csv', parse_dates=['timestamp'])
df['aqi_us'] = pd.to_numeric(df['aqi_us'], errors='coerce')
df = df.groupby('timestamp')['aqi_us'].mean().reset_index()
df.set_index('timestamp', inplace=True)
df = df.sort_index()
df = df.resample('H').mean().interpolate(method='linear')

def calculate_accuracy(y_true, y_pred):
    error = np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-5, None)
    accuracy = 100 - np.mean(error) * 100
    return accuracy

# Simplified parameter grid suitable for small dataset
param_grid = {
    'trend': [None, 'add', 'mul'],
    'smoothing_level': [0.1, 0.3, 0.5, 0.7, 0.9],  # Alpha values
    'smoothing_slope': [0.1, 0.3, 0.5, 0.7, 0.9]   # Beta values
}

# Train-test split
train_size = int(len(df) * 0.7)
train = df[:train_size]
test = df[train_size:]

# Grid Search
best_score = float('inf')
best_params = None
results = []

for trend in param_grid['trend']:
    for alpha in param_grid['smoothing_level']:
        for beta in param_grid['smoothing_slope']:
            try:
                with mlflow.start_run(run_name=f"ETS_{trend}_{alpha}_{beta}"):
                    # Log parameters
                    params = {
                        'trend': trend,
                        'smoothing_level': alpha,
                        'smoothing_slope': beta
                    }
                    mlflow.log_params(params)
                    
                    # Fit model
                    model = ExponentialSmoothing(
                        train['aqi_us'],
                        trend=trend,
                        seasonal=None  # Remove seasonal component
                    )
                    fitted_model = model.fit(
                        smoothing_level=alpha,
                        smoothing_slope=beta
                    )
                    
                    # Make predictions
                    forecast = fitted_model.forecast(len(test))
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(test['aqi_us'], forecast))
                    mae = mean_absolute_error(test['aqi_us'], forecast)
                    accuracy = calculate_accuracy(test['aqi_us'], forecast)
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "rmse": rmse,
                        "mae": mae,
                        "accuracy": accuracy
                    })
                    
                    results.append({
                        **params,
                        'rmse': rmse,
                        'mae': mae,
                        'accuracy': accuracy
                    })
                    
                    if rmse < best_score:
                        best_score = rmse
                        best_params = params
                        best_forecast = forecast
                    
                    print(f"ETS - Trend: {trend}, Alpha: {alpha}, Beta: {beta}")
                    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")
                    
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/output/ets_grid_search_results.csv', index=False)

# Train final model with best parameters
with mlflow.start_run(run_name="Best_ETS_Model"):
    # Log best parameters
    mlflow.log_params(best_params)
    
    # Fit final model
    final_model = ExponentialSmoothing(
        train['aqi_us'],
        trend=best_params['trend'],
        seasonal=None
    ).fit(
        smoothing_level=best_params['smoothing_level'],
        smoothing_slope=best_params['smoothing_slope']
    )
    
    # Make final predictions
    final_forecast = final_model.forecast(len(test))
    
    # Calculate final metrics
    final_rmse = np.sqrt(mean_squared_error(test['aqi_us'], final_forecast))
    final_mae = mean_absolute_error(test['aqi_us'], final_forecast)
    final_accuracy = calculate_accuracy(test['aqi_us'], final_forecast)
    
    # Log final metrics
    mlflow.log_metrics({
        "best_rmse": final_rmse,
        "best_mae": final_mae,
        "best_accuracy": final_accuracy
    })
    
    # Plot final results
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['aqi_us'], label='Actual', color='blue')
    plt.plot(test.index, final_forecast, label='Forecast', color='red')
    plt.title('Best ETS Model Forecast')
    plt.xlabel('Time')
    plt.ylabel('AQI')
    plt.legend()
    plt.tight_layout()
    
    # Save and log plot
    plt.savefig('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/images/best_ets_forecast.png')
    mlflow.log_artifact('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/images/best_ets_forecast.png')
    plt.close()

print("\nBest ETS Parameters:", best_params)
print(f"Best RMSE: {final_rmse:.2f}")
print(f"Best MAE: {final_mae:.2f}")
print(f"Best Accuracy: {final_accuracy:.2f}%")