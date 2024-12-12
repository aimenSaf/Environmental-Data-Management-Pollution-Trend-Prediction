@echo off
cd C:\Users\aimen\Desktop\mlops\course-project-aimenSaf
call C:\Users\aimen\Desktop\mlops\course-project-aimenSaf\venv\Scripts\activate.bat

:: Add changes to DVC
dvc add data\weather_data.csv data\air_quality_data.csv

:: Auto-stages DVC files, manually add other important files
git add data\*.dvc dvc_client_id_secret.json fetch_data.bat update_dvc_repo.bat code\script.py
git commit -m "Update data files with DVC and script adjustments"
dvc push
git push
