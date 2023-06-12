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

# Dataset
The IBM HR Analytics Employee Attrition Performance dataset was used which contains employee data for 1,470 employees with various information about the employees. It was used to predict when employees are going to quit by understanding the main drivers of employee churn. There are 35 factors included in the dataset. The dataset can be obtained at https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset.

As stated on the IBM website “This is a fictional data set created by IBM data scientists”. Its main purpose was to demonstrate the IBMWatson Analytics tool for employee attrition.

# Modelling

## Data pre-processing
The analysis begins with data preprocessing, which involved cleaning and formatting the dataset and converting the categorical variables into dummy variables. Machine Learning algorithms can typically only have numerical values as their predictor variables. Hence Label Encoding becomes necessary as they encode categorical labels with numerical values. To avoid introducing feature importance for categorical features with large numbers of unique values, Label Encoding and One-Hot Encoding were used.\\

The dataset used in this study has no missing values. In HR Analytics, it is unlikely for employee data to feature a large ratio of missing values since HR Departments typically have all personal and employment data on-file. However, the type of documentation data is being kept in (i.e. whether it is paper-based, Excel spreadsheets, databases, etc) has a massive impact on the accuracy and the ease of access to the HR data.\\

Feature scaling is an essential step in data preprocessing when developing predictive models. It involves transforming the numerical features in the dataset to a common scale or range. The main purpose of feature scaling is to ensure that all features contribute equally to the model training process, preventing any single feature from dominating the model's learning process due to differences in their original scales.\\

When features have significantly different scales, it becomes challenging to compare the importance or weight of each feature. Scaling the features to a common range enables fair comparison and interpretation of their contributions to the model's predictions.In this case the MinMaxScaler was used to essentially shrink the range of values such that it is now between 0 and 5.

## Experimental evaluations
In this study, we employed 10-fold cross-validation to evaluate the performance of the various models used for employee churn prediction. Cross-validation is a robust technique commonly used in machine learning to estimate the model's performance on unseen data and assess its generalization capabilities.
The 10-fold cross-validation procedure involves splitting the dataset into ten equal-sized subsets or "folds." The models are then trained and evaluated ten times, each time using a different fold as the validation set and the remaining nine folds as the training set. This process ensures that each data point is used for both training and validation, providing a comprehensive assessment of the model's performance across different subsets of the data.

By performing cross-validation, we mitigate the risk of overfitting or underfitting the model to a specific subset of the data. It helps us understand how the model generalizes to unseen instances and provides a more robust estimate of its performance.

The average performance metrics obtained from the 10-fold cross-validation provide a more reliable estimate of the model's performance than a single train-test split. Additionally, the standard deviation of the performance metrics across the folds provides insights into the consistency and stability of the model's predictions.

By utilizing 10-fold cross-validation, we ensure a rigorous evaluation of the models, capturing their performance across multiple subsets of the data. This approach enables us to make informed comparisons between different classifiers and select the model that exhibits the best overall performance in predicting employee churn.

# Results

| Model | Accuracy |
| -- | -- |
Random Forest | **85.12%** |
K - Nearest Neighbour | 84.67% |
Support Vector Machines | 84.30% |
Decision Tree Classifier | 80.31% |
Logistic Regression | 76.51% |
Gaussian Naive Bayes Classifier | 66.33% |

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
