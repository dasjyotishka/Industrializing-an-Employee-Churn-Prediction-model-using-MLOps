# Import mlflow
import mlflow
import mlflow.sklearn

from airflow import DAG
from airflow.operators.python import PythonOperator
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



df_sourcefile = pd.DataFrame()
df_HR = None
path='/home/dasjyotishka/airflow/Employee-Attrition.csv'


def read_data(**context):
	print("Inside read-data")

	global df_HR

	# Read Excel file
	df_sourcefile = pd.read_csv(path)
	print("Shape of dataframe is: {}".format(df_sourcefile.shape))

	# Make a copy of the original sourcefile
	df_HR = df_sourcefile.copy()

	"""Data Overview """

	# Dataset columns
	df_HR.columns

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

	# Store the dataframe in XCOM
	context['ti'].xcom_push(key='df_HR', value=df_HR)



def prep_encoding(**context):
	print("Inside prep_encoding")

	# Retrieve the dataframe from XCOM
	df_HR = context['ti'].xcom_pull(key='df_HR', task_ids='read_data')

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

	# Store the modified dataframe in XCOM
	context['ti'].xcom_push(key='df_HR', value=df_HR)
	
	



def prep_scale(**context):
	print("Inside prep_scale")

	# Retrieve the dataframe from XCOM
	df_HR = context['ti'].xcom_pull(key='df_HR', task_ids='prep_encoding')
	# Scale numerical features to range between 0 and 5
	scaler = MinMaxScaler(feature_range=(0, 5))
	HR_col = list(df_HR.columns)
	HR_col.remove('Attrition')
	for col in HR_col:
		df_HR[col] = df_HR[col].astype(float)
		df_HR[[col]] = scaler.fit_transform(df_HR[[col]])
	df_HR['Attrition'] = pd.to_numeric(df_HR['Attrition'], downcast='float')
	print('Size of Full Encoded Dataset: {}'.format(df_HR.shape))

	# Store the modified dataframe in XCOM
	context['ti'].xcom_push(key='df_HR', value=df_HR)



def split(df):
	# Assign the target to a new dataframe and convert it to a numerical feature
	target = df['Attrition'].copy()
	# target = pd.to_numeric(target, downcast='float')

	# Let's remove the target feature and redundant features from the dataset
	df.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1, inplace=True)
	print('Size of Full dataset is: {}'.format(df.shape))

	# Since we have class imbalance (i.e. more employees with turnover=0 than turnover=1)
	# Let's use stratify=y to maintain the same ratio as in the training dataset when splitting the dataset
	X_train, X_test, y_train, y_test = train_test_split(df,
													target,
													test_size=0.25,
													random_state=7,
													stratify=target)  
		
	print("Number transactions X_train dataset: ", X_train.shape)
	print("Number transactions y_train dataset: ", y_train.shape)
	print("Number transactions X_test dataset: ", X_test.shape)
	print("Number transactions y_test dataset: ", y_test.shape)
	
	return X_train, X_test, y_train, y_test



def ml_flow_function(**context):
	
		warnings.filterwarnings("ignore")
		np.random.seed(40)

		print("Inside mlflow function")
		experiment_id = "attrition_id"

		#Set tracker uri to enable logging the model as an artifact using http request
		mlflow.set_tracking_uri("http://127.0.0.1:5000")
		
		df_HR = context['ti'].xcom_pull(key='df_HR', task_ids='prep_scale')

		# Split the data into training and test sets. (0.75, 0.25) split.
		X_train, X_test, y_train, y_test = split(df_HR)

		# Create a simple Random forest classifier
		randForest = RandomForestClassifier(max_depth=7, random_state=0)
		randForest.fit(X_train, y_train)
		y_pred = randForest.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		print("Accuracy:", accuracy)
		
		f1 = f1_score(y_test, y_pred)
		print("F1 Score:", f1)

		# Log mlflow attributes for mlflow UI
		mlflow.log_param("accuracy", accuracy)
		mlflow.log_param("F1 score", f1)

		# Log the model with MLflow
		mlflow.sklearn.log_model(randForest, "employee_attrition_model")


with DAG(
    dag_id="employee_attrition_dag_2",
    start_date=datetime(2023, 5, 9),
    schedule_interval="@hourly",
    catchup=False
) as dag:

	read_data_task = PythonOperator(
		task_id="read_data",
		python_callable=read_data,
		provide_context=True
	)

	prep_encoding_task = PythonOperator(
		task_id="prep_encoding",
		python_callable=prep_encoding,
		provide_context=True
	)

	prep_scale_task = PythonOperator(
		task_id="prep_scale",
		python_callable=prep_scale,
		provide_context=True
	)

	ml_flow_function_task = PythonOperator(
		task_id="ml_flow_function",
		python_callable=ml_flow_function,
		provide_context=True
	)

	end_task = DummyOperator(task_id='end_task')

	read_data_task >> prep_encoding_task >> prep_scale_task >> ml_flow_function_task >> end_task


