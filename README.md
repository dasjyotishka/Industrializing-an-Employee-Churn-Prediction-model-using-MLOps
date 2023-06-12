# Implementing an Employee Churn Prediction Model using open source technologies

This project was the outcome of the Corporate Reseach Project as a part of Masters in Data Sciences and Business Analytics program at ESSEC-CentraleSupelec and monitored by Deloitte.

In this study, we propose and evaluate a predictive model for employee churn using machine learning techniques.Employee churn poses significant challenges to organizations in terms of productivity, knowledge loss, and increased recruitment costs. Predicting and mitigating
employee churn has become a critical aspect of human resource management. 

This project combined several of the most popular MLOps tools to showcase what the  workflow would look like using these tools, from experimentation to production. The experimentation loop uses Jupyter, MLflow, and Git. The production loop consists of Git, Airflow, MLflow, and FlaskAPI.

# Names of the students working on the project:

1. Jyotishka Das(jyotishka.das@essec.edu)
2. Xinran Yao(xinran.yao@essec.edu)
3. Priyam Dasgupta(priyam.dasgupta@essec.edu)
4. Shamir Mohamed(shamir.mohamed@essec.edu)
5. Jiayi Wu(jiayi.wu@essec.edu)

# Instructions for Execution:
Before the execution of the two python scripts in the repository (one for training and the other for testing), it is essential to have all the production infrastructure in the local system. Instructions to install all the dependencies and then to execute the scrips are mentioned below:
- **Download Windows Subsystem for Linux (WSL)**: Follow the steps as given in the URL https://www.freecodecamp.org/news/how-to-install-wsl2-windows-subsystem-for-linux-2-on-windows-10
- **Install Anaconda in the Linux subsystem**: Follow the steps as given in the URL https://learn.microsoft.com/en-us/windows/python/web-frameworks
- **Install MLFlow on WSL**: Follow the steps as given in the URL https://www.adaltas.com/en/2020/03/23/mlflow-open-source-ml-platform-tutorial/
- **Install Airflow on WSL**: Follow the steps as given in the URL https://www.freecodecamp.org/news/install-apache-airflow-on-windows-without-docker/
- **Execute the DAG to train and save the model**: Copy the main DAG code "implementing_employee_churn_model.py" to the DAG folder of the Airflow directory and start the Airflow server on http://localhost:8080/home. Now go to the Airflow UI of the server, and execute the DAG. The saved model would be registered in the MLFlow registry. The model can be registered through MLFlow UI on http://127.0.0.1:5000/.
- **Serve the model through Flask API**: Execute the python code "python_script_Flask_Testing.py" to serve the model through Flask API. To check whether the API is working, Postman can be used. In Postman, create a POST request on the address http://localhost:5001/predict. Under Headers, create a field "Content-Type" with value "multipart/form-data". Now create another field called "mlflow_run_id" with the value of the run-id of the saved MLFlow model that we want to use. Under body, in the src field, attach the csv file of the test-dataset whose output we want to see. Click on the "Send" button. If everything is perfect, then in the response field, a binary value corresponding to the attrition prediction results of every employees, along with the accuracy and F1 score of the model fetched from MLFlow registry would be displayed
