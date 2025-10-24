import os
import subprocess


print("Current working directory:", os.getcwd())

subprocess.run(['python', 'apis/app.py'])
subprocess.run(['python', 'models/ml_model/agri_price_predictor_ml.py'])

