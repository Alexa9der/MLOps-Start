"""
Last independent task:
 - replace BUCKET with your own;
 - Leave EXPERIMENT_NAME and DAG_ID as they are (references to the NAME variable);
 - connector names: pg_connection and s3_connection;
 - the data should be read from a table called california_housing;
 - data on S3 should be in the {NAME}/datasets/ and {NAME}/results/ folders.
"""
import json
import logging
import numpy as np
import pandas as pd
import pickle
import os

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Literal

import mlflow
from mlflow.models import infer_signature

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.hooks.base_hook import BaseHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Set up logging configuration
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

# Retrieve connection information for S3 from Airflow Connections
conn = BaseHook.get_connection("s3_connection")

# Set environment variables for AWS S3 access
os.environ["AWS_ACCESS_KEY_ID"] = conn.login
os.environ["AWS_SECRET_ACCESS_KEY"] = conn.password
os.environ["AWS_DEFAULT_REGION"] = conn.extra_dejson.get('region_name')
os.environ["MLFLOW_S3_ENDPOINT_URL"] = conn.extra_dejson.get('endpoint_url')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# TO-DO: Fill in your Telegram username
NAME = "Kanalosh_Oleksandr"

# TO-DO: Fill in your bucket name
BUCKET = "mlops-start"

EXPERIMENT_NAME = NAME
DAG_ID = NAME

# Define features and target variable for modeling
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"

# TO-DO: Create a dictionary of models
models = {
    "rf": RandomForestRegressor(),
    "lr": LinearRegression(),
    "hgb": HistGradientBoostingRegressor()
}


# Function to initialize metrics and MLflow experiment
def init() -> Dict[str, Any]:
    # Log initialization step
    _LOG.info("Initialization.")

    # Initialize metrics dictionary
    metrics = {}
    
    # Step 1: Collect start timestamp, run_id, experiment_name, experiment_id
    metrics["start_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Start timestamp collected.")

    # Check if the experiment exists in MLflow, if not create a new one
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        _LOG.info(f"Creating new experiment: {EXPERIMENT_NAME}")
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f"s3://{BUCKET}/mlflow")
    else:
        _LOG.info(f"Experiment {EXPERIMENT_NAME} already exists.")
        experiment_id = experiment.experiment_id
        
    # Set the current experiment in MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Update metrics with experiment name and ID
    metrics["experiment_name"] = EXPERIMENT_NAME
    metrics["experiment_id"] = experiment_id
    
    _LOG.info(f"Experiment set with ID: {experiment_id}")
    
    # Step 3: Create a parent run in MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name="Parent_Run") as parent_run:
        metrics['run_id'] = parent_run.info.run_id
        
    _LOG.info("Initialization end.")

    return metrics
    

def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
    _LOG.info("Starting data extraction from PostgreSQL.")

    # Step 1: Initialize metrics and collect data_download_start timestamp
    ti: TaskInstance = kwargs["ti"]
    metrics: Dict[str, Any] = ti.xcom_pull(task_ids="init")
    metrics["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Data download start timestamp collected.")

    # Step 2: Establish connections
    pg_hook = PostgresHook(postgres_conn_id="pd_connection")
    con = pg_hook.get_conn()
    _LOG.info("Connected to PostgreSQL.")

    s3_hook = AwsHook(aws_conn_id="s3_connection")
    session = s3_hook.get_session(region_name="eu-north-1")
    resource = session.resource("s3")
    _LOG.info("Connected to S3.")

    # Step 3: Read data from PostgreSQL
    data = pd.read_sql_query("SELECT * FROM california_housing", con)
    _LOG.info("Data read from PostgreSQL.")

    # Step 4: Save data to S3 in pickle format under {NAME}/datasets/
    file_name = f"{NAME}/datasets/california_housing.pkl"
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, file_name).put(Body=pickle_byte_obj)
    _LOG.info("Data saved to S3.")

    # Step 5: Update metrics with data_download_end timestamp
    metrics["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Data download end timestamp logged.")

    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    _LOG.info("Starting data preparation.")
    
    # Step 1: Initialize metrics and collect data_preparation_start timestamp
    ti: TaskInstance = kwargs["ti"]
    metrics: Dict[str, Any] = ti.xcom_pull(task_ids="get_data_from_postgres")
    metrics["data_preparation_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Data preparation start timestamp collected.")
    
    # Step 2: Establish S3 connection
    s3_hook = AwsHook(aws_conn_id="s3_connection")
    _LOG.info("Connected to S3.")

    # Step 3: Read data from S3
    file_name = f"{NAME}/datasets/california_housing.pkl"
    file = s3_hook.download_file(key=file_name, bucket_name=BUCKET)
    data = pd.read_pickle(file)
    X, y = data[FEATURES], data[TARGET]
    _LOG.info("Data read from S3.")

    # Step 4: Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _LOG.info("Data split into train and test sets.")

    # Step 5: Perform preprocessing using StandardScaler
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)
    _LOG.info("Data preprocessing completed using StandardScaler.")

    # Step 6: Prepare 4 processed datasets
    names = "X_train_fitted", "X_test_fitted", "y_train", "y_test"
    datas = X_train_fitted, X_test_fitted, y_train, y_test
    
    # Save data to S3 in {NAME}/datasets/ folder
    session = s3_hook.get_session(region_name="eu-north-1")
    resource = session.resource("s3")
    
    for name, data in zip(names, datas):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET, f"{NAME}/datasets/{name}.pkl").put(Body=pickle_byte_obj)
    _LOG.info("Processed datasets saved to S3.")

    # Finalize metrics with data_preparation_end timestamp
    metrics["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("End of data preparation.")

    return metrics


def train_mlflow_model(model: Any, name: str, X_train: np.ndarray,
                       X_test: np.ndarray, y_train: pd.Series,
                       y_test: pd.Series) -> None:
    """
    Train a machine learning model, make predictions, and log results using MLFlow.

    Parameters:
    - model (Any): The machine learning model object.
    - name (str): The name to associate with the logged model in MLFlow.
    - X_train (np.ndarray): Training features.
    - X_test (np.ndarray): Test features.
    - y_train (pd.Series): Training target values.
    - y_test (pd.Series): Test target values.
    """
    
    # TO-DO 1: Train the model.
    model.fit(X_train, y_train)
    
    # TO-DO 2: Make predictions.
    prediction = model.predict(X_test)
    
    # TO-DO 3: Save training results using MLFlow.
    # Infer signature from input and output data
    signature = infer_signature(X_test, prediction)
    
    # Log the trained model in MLFlow
    model_info = mlflow.sklearn.log_model(model, name, signature=signature)
    
    # Evaluate the model and log metrics
    mlflow.evaluate(
        model_info.model_uri,  # URI of the logged model
        data=X_test,           # Test data for evaluation
        targets=y_test.values, # Target values for evaluation
        model_type="regressor",# Model type (e.g., "regressor", "classifier")
        evaluators=["default"] # List of evaluators to apply
    )

def train_model(**kwargs) -> Dict[str, Any]:
    """
    Function to train a machine learning model, log training metrics using MLFlow,
    and return the collected metrics as a dictionary.

    Parameters:
    - kwargs (dict): Keyword arguments containing Airflow task instance (`ti`) and others.

    Returns:
    - metrics (dict): Dictionary containing metrics collected during the training process.
    """
    
    # TO-DO 1: Collect metrics for this step
    model_name = kwargs['model_name']
    _LOG.info(f"Training model: {model_name}")
    
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="prepare_data")
    metrics[f"train_start_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    # TO-DO 2: Create connectors
    s3_hook = S3Hook("s3_connection")
    _LOG.info("Connected to S3.")
    
    # TO-DO 3: Read data from S3
    names = "X_train_fitted", "X_test_fitted", "y_train", "y_test"
    data = {}
    for name in names:
        file = s3_hook.download_file(key=f"{NAME}/datasets/{name}.pkl", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file).copy()

    # TO-DO 4: Train models and log training using MLFlow
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    
    with mlflow.start_run(run_name=f"{model_name}_Training", experiment_id=experiment_id, nested=True) as child_run:
        
        mlflow.set_tags({"mlflow.parentRunId": metrics['run_id']})
        
        # Call function to train and log the model using MLFlow
        train_mlflow_model(models[model_name], model_name, data["X_train_fitted"], data["X_test_fitted"], data["y_train"], data["y_test"])
        
        _LOG.info(f"Run completed for model: {model_name}")

    # Update metrics with end timestamp for this step
    metrics[f"train_end_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info(f"Training for model {model_name} completed.")

    return metrics


def save_results(**kwargs) -> None:
    """
    Function to save training metrics to S3 in JSON format.

    Parameters:
    - kwargs (dict): Keyword arguments containing Airflow task instance (`ti`).

    Returns:
    - None
    """
    
    _LOG.info("Beginning save metrics.")
    
    # TO-DO 1: Collect end timestamp
    ti = kwargs["ti"]
    result = {}
    
    # Loop through each model to collect metrics
    for model_name in models.keys():
        model_metrics = ti.xcom_pull(task_ids=f"train_{model_name}")
        result.update(model_metrics)
    
    # Add end timestamp to the collected metrics
    end_timestamp = datetime.now().strftime("%Y_%m_%d_%H")
    result["end_timestamp"] = end_timestamp
    
    # Convert metrics to JSON format
    json_byte_object = json.dumps(result)
    
    # TO-DO 2: Save training results to S3 in {NAME}/results/{date}.json file
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")
    _LOG.info("Connected to S3.")
    
    file_name = f"{NAME}/results/{end_timestamp}.json"
    resource.Object(BUCKET, file_name).put(Body=json_byte_object)
    
    _LOG.info("End save metrics.")




#################################### INIT DAG ####################################

# TO-DO: Fill in with your data: configure the owner and retries policy.
default_args = {
    "owner": "Oleksandr Kanalosh",
    "email": ["oleksandrairflow@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5) 
}

# TO-DO: Fill in the remaining parameters.
dag = DAG(
          dag_id=DAG_ID,
          default_args=default_args,
          schedule_interval="0 22 * * 5",  
          start_date=days_ago(2),
          catchup=False,
          tags=["mlops"],  
          )

task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id="get_data_from_postgres",
                               python_callable=get_data_from_postgres,
                               dag=dag,
                               provide_context=True)

task_prepare_data = PythonOperator(task_id="prepare_data",
                                   python_callable=prepare_data,
                                   dag=dag,
                                   provide_context=True)

training_model_tasks = [
    PythonOperator(task_id=f"train_{model_name}",
                   python_callable=train_model,
                   dag=dag,
                   provide_context=True,
                   op_kwargs={"model_name": model_name})
    for model_name in models.keys()
]

task_save_results = PythonOperator(task_id="save_results",
                                   python_callable=save_results,
                                   dag=dag,
                                   provide_context=True)

# TO-DO: Define the DAG architecture.
task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results

