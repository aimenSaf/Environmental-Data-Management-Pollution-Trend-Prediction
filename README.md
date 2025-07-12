# 🌍 Environmental Data Management & Pollution Trend Prediction
---

## 📌 Overview

This project implements a complete MLOps pipeline to collect live environmental data, version it using DVC, build forecasting models to predict pollution trends, and deploy the results through a simple Flask web app. 

The project is divided into two main tasks:

- **Task 1:** Environmental Data Collection & Management using DVC  
- **Task 2:** Pollution Trend Prediction using MLflow & Model Deployment

---

## 🧪 Task 1: Managing Environmental Data with DVC

### ✅ Pre-requisites

- Python environment setup:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  pip install -r requirements.txt



- APIs used:

  OpenWeatherMap
  
  IQAir (AirVisual)


### 🔨 Setup Steps
1. DVC Initialization
   ```bash
   dvc init
   git add .dvc .gitignore
   git commit -m "Initialize DVC repository"

2. Remote Storage with Google Drive
   ```bash
   pip install "dvc[gdrive]"
   dvc remote add -d myremote gdrive://<folder-id>

3. Environment Variables
   Create a .env file with the following content:
     ```bash
     OPENWEATHERMAP_API_KEY=your_openweathermap_key
     AIRVISUAL_API_KEY=your_airvisual_key
4. And install python-dotenv:
   ```bash
   pip install python-dotenv
   
5. Data Collection Script
   - Fetches data from both APIs
   - Appends data to CSV files in the data/ directory
   - Logs actions in data_collection.log

6. Version Control with DVC
   ```bash
   dvc add data/weather_data.csv
   dvc add data/air_quality_data.csv
   git add data/*.dvc
   git commit -m "Track data files with DVC"
   dvc push

7. Automation
   - Created a .bat script to periodically run the data collection script
   - Scheduled using Windows Task Scheduler

---

## 📊 Task 2: Pollution Trend Prediction with MLflow
### 🧹 Data Preparation
- Merged both datasets into output/combined_data.csv
- Tracked the merged file with DVC:
  ```bash
  dvc add output/combined_data.csv
  git add output/*.dvc
  git commit -m "Add combined data file to DVC"
  dvc push

### 📈 Model Development
- Models used:
  -- ARIMA (AutoRegressive Integrated Moving Average)
  -- ETS (Error, Trend, Seasonality)
- Trained models using separate scripts:
  ```bash
  python arima_mlflow.py
  python ets_mlflow.py
  mlflow ui

### 🧪 Hyperparameter Tuning
- Tuned parameters for ARIMA and ETS
- Tracked results using MLflow UI

### 📊 Evaluation
- Compared model performance using MLflow metrics
- Best Model: ETS with 97.61% accuracy

---
## 🚀 Deployment
- Built a simple Flask app
- Accepts number of days as input
- Predicts future AQI values and displays them by date

---

## 📁 Directory Structure
- ```bash
  project/
  │
  ├── data/
  │   ├── weather_data.csv│   └── air_quality_data.csv
  │
  ├── output/
  │   └── combined_data.csv
  │
  ├── scripts/
  │   ├── data_collection.py
  │   ├── arima_mlflow.py
  │   └── ets_mlflow.py
  │
  ├── app.py                 # Flask deployment script
  ├── update_data.bat        # Automates data fetch and push
  ├── .env                   # API keys (not tracked by Git)
  ├── requirements.txt
  ├── README.md
  └── mlruns/                # MLflow tracking




---

## 📦 Requirements
- ```bash
  pip install requests numpy pandas matplotlib scikit-learn dvc mlflow flask python-dotenv

## 🔐 Notes
- Keep your .env file private and exclude it from Git.

- Ensure Google Drive authentication is completed for DVC.

- Maintain a clean .gitignore to avoid tracking sensitive or large data files.

