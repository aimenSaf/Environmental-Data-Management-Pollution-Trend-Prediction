@echo off
cd C:\Users\aimen\Desktop\mlops\course-project-aimenSaf
call C:\Users\aimen\Desktop\mlops\course-project-aimenSaf\venv\Scripts\activate.bat
py C:\Users\aimen\Desktop\mlops\course-project-aimenSaf\code\script.py
dvc add data\weather_data.csv data\air_quality_data.csv
git add .
git commit -m "Update data files"
dvc push
git push
