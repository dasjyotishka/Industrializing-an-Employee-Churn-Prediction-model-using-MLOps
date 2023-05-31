from flask import Flask, request, jsonify

from datetime import datetime


# Import mlflow
import mlflow
import mlflow.sklearn

#from airflow import DAG
#from airflow.operators.python import PythonOperator
from datetime import datetime

from airflow.models import XCom

from airflow.operators.dummy import DummyOperator

# importing libraries for data handling and analysis
import json

import pendulum

#from airflow.decorators import dag, task


import warnings
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
#from IPython.display import display
pd.options.display.max_columns = None
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split  # import 'train_test_split'
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Libraries for data modelling
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Common sklearn Model Helpers
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# sklearn modules for performance metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# importing misceallenous libraries
import os
import re
import sys
import timeit
import string
from datetime import datetime
from time import time
from dateutil.parser import parse
app = Flask(__name__)



def make_predictions(file):
	#file = request.files['file']
	if file:
		df = pd.read_csv(file)
		# Perform data preprocessing and model prediction using mlflow
		# Make a copy of the original sourcefile
		df_HR = df.copy()

		"""Data Overview """

		# Dataset header
		df_HR.head()

		"""The dataset contains several numerical and categorical columns providing various information on employee's personal and employment details."""

		df_HR.columns.to_series().groupby(df_HR.dtypes).groups

		# Columns datatypes and missign values
		df_HR.info()

		"""
		The data provided has no missing values. 
		In HR Analytics, employee data is unlikely to feature large ratio of missing values as HR Departments typically have all personal and employment data on-file. However, the type of documentation data is being kept in (i.e. whether it is paper-based, Excel spreadhsheets, databases, etc) has a massive impact on the accuracy and the ease of access to the HR data.
		"""

		#Numerical features overview

		df_HR.describe()

		print("Prep_encoding starts")



		# Create a label encoder object
		le = LabelEncoder()

		# Label Encoding will be used for columns with 2 or less unique values
		le_count = 0
		for col in df_HR.columns[1:]:
			if df_HR[col].dtype == 'object':
				if len(list(df_HR[col].unique())) <= 2:
					le.fit(df_HR[col])
					df_HR[col] = le.transform(df_HR[col])
					le_count += 1
		print('{} columns were label encoded.'.format(le_count))

		# convert rest of categorical variable into dummy
		df_HR = pd.get_dummies(df_HR, drop_first=True)


		print("Inside prep_scale")


		# Scale numerical features to range between 0 and 5
		scaler = MinMaxScaler(feature_range=(0, 5))
		HR_col = list(df_HR.columns)
		HR_col.remove('Attrition')
		for col in HR_col:
			df_HR[col] = df_HR[col].astype(float)
			df_HR[[col]] = scaler.fit_transform(df_HR[[col]])
		df_HR['Attrition'] = pd.to_numeric(df_HR['Attrition'], downcast='float')
		print('Size of Full Encoded Dataset: {}'.format(df_HR.shape))


		#Test blocks
		warnings.filterwarnings("ignore")
		np.random.seed(40)

		print("Inside mlflow function")
		experiment_id = "attrition_id"

		mlflow.set_tracking_uri("http://127.0.0.1:5000")



		# let's remove the target feature and redundant features from the dataset
		df_HR.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1, inplace=True)
		print('Size of Full dataset is: {}'.format(df_HR.shape))




		# Test a simple Random forest classifier
		logged_model = 'runs:/c1fd3d0c5f364a6cad5c5b459f87adc9/employee_attrition_model'

		# Load model as a PyFuncModel.
		loaded_model = mlflow.pyfunc.load_model(logged_model)

		# Predict on a Pandas DataFrame.

		y_pred = loaded_model.predict(df_HR)

		# Retrieve the F1 score and accuracy from the model parameters

		# Fetch the MLflow run associated with the model
		run = mlflow.get_run(loaded_model.metadata.run_id)

		# Retrieve the F1 score from the run's metrics or artifacts
		f1_score = run.data.metrics.get("F1 score")
		if f1_score is None:
			f1_score = run.data.params.get("F1 score")


		# Retrieve the accuracy from the run's metrics or artifacts
		accuracy = run.data.metrics.get("accuracy")
		if accuracy is None:
			accuracy = run.data.params.get("accuracy")


		# Convert the NumPy array to a list
		y_pred = y_pred.tolist()

		print("y_pred is:")
		print(y_pred)


		# Return the predictions as JSON response
		response = {
        "predictions": y_pred,
        "f1_score": f1_score,
		"accuracy": accuracy
    }
		return jsonify(response)
	else:
		return jsonify({'error': 'Invalid file format. Please upload a CSV file.'})

@app.route('/predict', methods=['POST'])
def handle_predict():
    if 'src' in request.files:
        file = request.files['src']
        file_path = os.path.join('/home/dasjyotishka/airflow/Test_files_repository', file.filename)
        file.save(file_path)
        return make_predictions(file_path)
    else:
        return jsonify({'error': 'No file provided.'})

if __name__ == '__main__':
    host = '127.0.0.1'  # Specify the desired host IP address
    port = 5001  # Specify the desired port number
    app.run(host=host, port=port)

