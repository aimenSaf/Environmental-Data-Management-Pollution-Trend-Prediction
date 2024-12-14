# evaluation.py
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

def get_best_runs():
    # Get the best run from ARIMA experiment
    arima_exp = mlflow.get_experiment_by_name("AQI_Forecasting_ARIMA_Tuning")
    arima_runs = mlflow.search_runs(experiment_ids=[arima_exp.experiment_id])
    best_arima_run = arima_runs.loc[arima_runs['metrics.rmse'].idxmin()]
    
    # Get the best run from ETS experiment
    ets_exp = mlflow.get_experiment_by_name("AQI_Forecasting_ETS_Tuning")
    ets_runs = mlflow.search_runs(experiment_ids=[ets_exp.experiment_id])
    best_ets_run = ets_runs.loc[ets_runs['metrics.rmse'].idxmin()]
    
    return best_arima_run, best_ets_run

def get_model_params(run):
    params = {}
    for col in run.index:
        if col.startswith('params.'):
            params[col.replace('params.', '')] = run[col]
    return params

def compare_models():
    best_arima_run, best_ets_run = get_best_runs()
    
    # Create comparison DataFrame
    comparison_data = {
        'Metric': ['RMSE', 'MAE', 'Accuracy'],
        'ARIMA': [
            best_arima_run['metrics.rmse'],
            best_arima_run['metrics.mae'],
            best_arima_run['metrics.accuracy']
        ],
        'ETS': [
            best_ets_run['metrics.rmse'],
            best_ets_run['metrics.mae'],
            best_ets_run['metrics.accuracy']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    x = ['RMSE', 'MAE']
    arima_metrics = [best_arima_run['metrics.rmse'], best_arima_run['metrics.mae']]
    ets_metrics = [best_ets_run['metrics.rmse'], best_ets_run['metrics.mae']]
    
    x_pos = range(len(x))
    width = 0.35
    
    plt.bar([p - width/2 for p in x_pos], arima_metrics, width, label='ARIMA', color='blue', alpha=0.7)
    plt.bar([p + width/2 for p in x_pos], ets_metrics, width, label='ETS', color='red', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Model Comparison')
    plt.xticks(x_pos, x)
    plt.legend()
    plt.savefig('images/model_comparison.png')
    plt.close()
    
    # Determine best model
    if best_arima_run['metrics.rmse'] < best_ets_run['metrics.rmse']:
        best_model = 'ARIMA'
        best_run = best_arima_run
    else:
        best_model = 'ETS'
        best_run = best_ets_run
    
    # Get parameters for the best model
    best_params = get_model_params(best_run)
    
    print(f"\nBest Model: {best_model}")
    print(f"Best Model Parameters: {best_params}")
    print(f"Best Model Metrics:")
    print(f"RMSE: {best_run['metrics.rmse']:.2f}")
    print(f"MAE: {best_run['metrics.mae']:.2f}")
    print(f"Accuracy: {best_run['metrics.accuracy']:.2f}%")
    
    # Save results to a file
    with open('best_model_info.txt', 'w') as f:
        f.write(f"Best Model: {best_model}\n")
        f.write(f"Best Model Parameters: {best_params}\n")
        f.write(f"RMSE: {best_run['metrics.rmse']:.2f}\n")
        f.write(f"MAE: {best_run['metrics.mae']:.2f}\n")
        f.write(f"Accuracy: {best_run['metrics.accuracy']:.2f}%\n")
    
    return best_model, best_params, best_run

if __name__ == "__main__":
    compare_models()