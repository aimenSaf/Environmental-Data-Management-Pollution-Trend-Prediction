import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import mlflow
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("AQI_Forecasting_ARIMA_Tuning")

# Load and preprocess data
df = pd.read_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/output/combined_data.csv', parse_dates=['timestamp'])
df['aqi_us'] = pd.to_numeric(df['aqi_us'], errors='coerce')
df = df.groupby('timestamp')['aqi_us'].mean().reset_index()
df.set_index('timestamp', inplace=True)
df = df.sort_index()
df = df.resample('H').mean().interpolate(method='linear')
df = df.dropna()

def calculate_accuracy(y_true, y_pred):
    error = np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-5, None)
    accuracy = 100 - np.mean(error) * 100
    return accuracy

# Define parameter grid
param_grid = {
    'p': range(0, 4),  # AR order
    'd': range(0, 3),  # Differencing
    'q': range(0, 4),  # MA order
}

# Create all combinations of parameters
param_combinations = list(itertools.product(
    param_grid['p'], 
    param_grid['d'], 
    param_grid['q']
))

# Train-test split
train_size = int(len(df) * 0.7)
train = df[:train_size]
test = df[train_size:]

# Grid Search with Cross-validation
best_score = float('inf')
best_params = None
results = []

for params in param_combinations:
    p, d, q = params
    try:
        with mlflow.start_run(run_name=f"ARIMA_{p}_{d}_{q}"):
            # Log parameters
            mlflow.log_params({
                "p": p,
                "d": d,
                "q": q
            })
            
            # Fit model
            model = ARIMA(train['aqi_us'], order=(p, d, q))
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=len(test))
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(test['aqi_us'], forecast))
            mae = mean_absolute_error(test['aqi_us'], forecast)
            accuracy = calculate_accuracy(test['aqi_us'], forecast)
            aic = fitted_model.aic
            bic = fitted_model.bic
            
            # Log metrics
            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "accuracy": accuracy,
                "aic": aic,
                "bic": bic
            })
            
            results.append({
                'params': params,
                'rmse': rmse,
                'mae': mae,
                'accuracy': accuracy,
                'aic': aic,
                'bic': bic
            })
            
            # Update best parameters if current model is better
            if rmse < best_score:
                best_score = rmse
                best_params = params
                
            print(f"ARIMA{params} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")
            
    except Exception as e:
        print(f"Error with parameters {params}: {str(e)}")
        continue

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/output/arima_grid_search_results_tuned.csv', index=False)

# Train final model with best parameters
with mlflow.start_run(run_name="Best_ARIMA_Model"):
    p, d, q = best_params
    model = ARIMA(train['aqi_us'], order=(p, d, q))
    final_model = model.fit()
    forecast = final_model.forecast(steps=len(test))
    
    # Calculate final metrics
    final_rmse = np.sqrt(mean_squared_error(test['aqi_us'], forecast))
    final_mae = mean_absolute_error(test['aqi_us'], forecast)
    final_accuracy = calculate_accuracy(test['aqi_us'], forecast)
    
    # Log best parameters and metrics
    mlflow.log_params({
        "best_p": p,
        "best_d": d,
        "best_q": q
    })
    mlflow.log_metrics({
        "best_rmse": final_rmse,
        "best_mae": final_mae,
        "best_accuracy": final_accuracy
    })
    
    # Plot final results
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['aqi_us'], label='Actual', color='blue')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title(f'Best ARIMA Model Forecast (p={p}, d={d}, q={q})')
    plt.xlabel('Time')
    plt.ylabel('AQI')
    plt.legend()
    plt.tight_layout()
    
    # Save and log plot
    plt.savefig('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/images/best_arima_forecast.png')
    mlflow.log_artifact('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/images/best_arima_forecast.png')
    plt.close()

print("\nBest ARIMA Parameters:", best_params)
print(f"Best RMSE: {final_rmse:.2f}")
print(f"Best MAE: {final_mae:.2f}")
print(f"Best Accuracy: {final_accuracy:.2f}%")