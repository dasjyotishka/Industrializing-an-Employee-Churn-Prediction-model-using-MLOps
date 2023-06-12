# Implementing an Employee Churn Prediction Model using open source technologies

This project was the outcome of the Corporate Reseach Project as a part of Masters in Data Sciences and Business Analytics program at ESSEC-CentraleSupelec and monitored by Deloitte.

In this study, we propose and evaluate a predictive model for employee churn using machine learning techniques. The objective of this study was to develop an accurate and robust employee churn prediction model that can assist organizations in identifying employees at risk of leaving. The model utilizes a comprehensive dataset comprising various employee attributes, such as demographic information, job-related factors, performance metrics, and engagement indicators. The IBM HR Analytics Employee Attrition & Performance dataset was used for evaluating the model performance. Machine learning algorithms such as Random Forest and Logistic Regression were implemented. The performance of these classifiers was evaluated based on metrics such as accuracy, precision, recall, and F1-score. Additionally, feature importance analysis was conducted to identify the top influential factors contributing to employee churn which would enable organizations to focus their retention strategies on these critical areas.

The developed model can serve as a valuable tool for HR professionals and decision-makers in proactively managing employee retention efforts.However, the utility of the model is minimized unless it can be productionized in a large scale industrial setup. Industrialization involves deploying and scaling it to production environments, which requires reliable and efficient processes for managing and monitoring the ML workflow. These ML pipelines which were established using an open source MLOps platform were ensured to be reliable, efficient, easy to manage, and reusable to any project scenario, thereby cutting future R&D costs and increasing efficiency.

The primary objective of this project is to facilitate the industrialization of a specific use case by establishing a comprehensive machine learning (ML) pipeline. This entails undertaking various tasks, including the development of ML models, orchestration of the pipeline, and implementing model serving capabilities to cater to business requirements. The project will delve into the construction of the pipeline by leveraging a diverse range of cloud technologies, such as MLflow and Airflow, which offer powerful features for managing and deploying ML workflows.

The selected use case for this endeavor revolves around predicting employee churn rate, a crucial aspect that holds significant implications for the human resource department within the organization. By implementing advanced ML techniques, this project aims to generate valuable insights and forecasts, enabling the HR department to proactively address employee attrition and optimize workforce management strategies. Through the integration of open source technologies and the construction of the MLOps pipeline, the project seeks to enhance decision-making processes, foster efficiency, and ultimately contribute to the long-term success of the company.

This project combined several of the most popular MLOps tools to showcase what the  workflow would look like using these tools, from experimentation to production. The experimentation loop uses Jupyter, MLflow, and Git. The production loop consists of Git, Airflow, MLflow, and FlaskAPI.

# Names of the students working on the project:

1. Jyotishka Das(jyotishka.das@essec.edu)
2. Xinran Yao(xinran.yao@essec.edu)
3. Priyam Dasgupta(priyam.dasgupta@essec.edu)
4. Shamir Mohamed(shamir.mohamed@essec.edu)
5. Jiayi Wu(jiayi.wu@essec.edu)

# Overview of the MLOps pipelines developed

MLOps, which stands for Machine Learning Operations, refers to the practices and techniques used to streamline and operationalize machine learning models in production environments that focus on the challenges of deploying, monitoring, and maintaining machine learning systems. It helps establish a robust and efficient workflow that encompasses the entire lifecycle of a machine learning project. \\

Secondly, MLOps focuses on building scalable and automated pipelines for data preprocessing, model training, and evaluation. These pipelines enable efficient data ingestion, feature engineering, model training, hyperparameter tuning, and evaluation. Automation helps reduce manual errors, enhances productivity, and enables rapid experimentation and iteration. Another critical aspect of MLOps is model deployment and serving. It involves packaging trained models into deployable formats, setting up scalable and robust infrastructure for serving predictions, and monitoring the model's performance in real-world scenarios. In this specific use case, MLOps technologies like Airflow and MLflow were used to provide a framework for managing the entire ML workflow, from data preparation to model training, evaluation, and deployment. An entire overview of the scalable MLOps pipelines that was created for this project is shown in the figure belo. The red-dotted lines show the pipelines which were not yet created for this project but can be taken up in the future for more versatility.

![image](https://github.com/dasjyotishka/Implementing-an-Employee-Churn-Prediction-model-using-open-source-technologies/assets/55792433/7ae752f8-be48-4347-b5b6-59fd976d5c45)


# Instructions for Execution:
Before the execution of the two python scripts in the repository (one for training and the other for testing), it is essential to have all the production infrastructure in the local system. Instructions to install all the dependencies and then to execute the scrips are mentioned below:
- **Download Windows Subsystem for Linux (WSL)**: Follow the steps as given in the URL https://www.freecodecamp.org/news/how-to-install-wsl2-windows-subsystem-for-linux-2-on-windows-10
- **Install Anaconda in the Linux subsystem**: Follow the steps as given in the URL https://learn.microsoft.com/en-us/windows/python/web-frameworks
- **Install MLFlow on WSL**: Follow the steps as given in the URL https://www.adaltas.com/en/2020/03/23/mlflow-open-source-ml-platform-tutorial/
- **Install Airflow on WSL**: Follow the steps as given in the URL https://www.freecodecamp.org/news/install-apache-airflow-on-windows-without-docker/
- **Execute the DAG to train and save the model**: Copy the main DAG code "implementing_employee_churn_model.py" to the DAG folder of the Airflow directory and start the Airflow server on http://localhost:8080/home. Now go to the Airflow UI of the server, and execute the DAG. The saved model would be registered in the MLFlow registry. The model can be registered through MLFlow UI on http://127.0.0.1:5000/.
- **Serve the model through Flask API**: Execute the python code "python_script_Flask_Testing.py" to serve the model through Flask API. To check whether the API is working, Postman can be used. In Postman, create a POST request on the address http://localhost:5001/predict. Under Headers, create a field "Content-Type" with value "multipart/form-data". Now create another field called "mlflow_run_id" with the value of the run-id of the saved MLFlow model that we want to use. Under body, in the src field, attach the csv file of the test-dataset whose output we want to see. Click on the "Send" button. If everything is perfect, then in the response field, a binary value corresponding to the attrition prediction results of every employees, along with the accuracy and F1 score of the model fetched from MLFlow registry would be displayed
