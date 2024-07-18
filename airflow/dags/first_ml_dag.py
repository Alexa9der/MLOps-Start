import io
import json
import numpy as np
import pandas as pd
import pickle
import logging

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


# Setting up logging configuration
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

# Default arguments for Airflow tasks
DEFAULT_ARGS = {
    "owner": "Oleksandr Kanalosh",
    "email": ["oleksandrairflow@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5) 
}

# Define the DAG object
first_ml_dag = DAG(
    dag_id="first_ml_dag",
    schedule_interval="0 1 * * *",  # Run daily at 1:00 AM
    start_date=days_ago(2),  # Start 2 days ago
    catchup=False,  # Don't backfill past DAG runs
    tags=["mlops"],  # Tags for identifying the DAG's purpose
    default_args=DEFAULT_ARGS  # Default arguments for tasks in the DAG
)

# S3 bucket and folder constants
FIRST_BUCKET = "first-airflou-test"
ROOT_FOLDER = "First_action"
RAW_DATA = "First_action/california_housing.pkl"

# Features and target variable for model training
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"

# Function to initialize the training pipeline
def init() -> None:
    _LOG.info("Train pipeline started.")

# Function to retrieve data from PostgreSQL and store it in S3
def get_data_from_postgres() -> None:
    try:
        # Connect to PostgreSQL and retrieve data
        pg_hook = PostgresHook("pd_connection")
        con = pg_hook.get_conn()
        data = pd.read_sql_query("SELECT * FROM california_housing", con)
        
        # Connect to S3 and save data as a pickle file
        s3_hook = S3Hook("s3_connection")
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(FIRST_BUCKET, RAW_DATA).put(Body=pickle_byte_obj)
        
        _LOG.info("Data download finished.")
        
    except Exception as e:
        _LOG.error(f"Error in data download and storage: {str(e)}")
        raise e

# Function to prepare data for model training
def prepare_data() -> None:
    data = None 
    
    try:
        # Connect to S3 and download prepared datasets
        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(key=RAW_DATA, bucket_name=FIRST_BUCKET)
        data = pd.read_pickle(file)

        # Check if required features and target are available in the dataset
        if any(feature not in data.columns for feature in FEATURES) or TARGET not in data.columns:
            raise ValueError(f"Features ({FEATURES}) or Target ({TARGET}) columns are missing in the dataset.")

        # Split data into train and test sets, perform scaling
        X, y = data[FEATURES].copy(), data[TARGET].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        # Save prepared datasets back to S3
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")
        names = "X_train", "X_test", "y_train", "y_test"
        datas = X_train, X_test, y_train, y_test

        for name, data in zip(names, datas):
            pickle_byte_obj = pickle.dumps(data)
            resource.Object(FIRST_BUCKET, f"{ROOT_FOLDER}/prepared_datasets/{name}.pkl").put(Body=pickle_byte_obj)
            
        _LOG.info("Data preparation finished.")
        
    except Exception as e:
        if data is None:
            _LOG.error(f"Error in data preparation: {str(e)}")
        else:
            _LOG.error(f"Error in data preparation after loading data: {str(e)}")
        raise e

# Function to train a model using prepared data
def train_model() -> None:
    try:
        # Connect to S3 and download prepared datasets
        s3_hook = S3Hook("s3_connection")
        names = "X_train", "X_test", "y_train", "y_test"
        data = {}
        for name in names:
            file = s3_hook.download_file(key=f"{ROOT_FOLDER}/prepared_datasets/{name}.pkl", bucket_name=FIRST_BUCKET)
            data[name] = pd.read_pickle(file)

        # Train a RandomForestRegressor model
        model = RandomForestRegressor()
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])

        # Calculate evaluation metrics
        result = {}
        result["mean_squared_error"] = mean_squared_error(data["y_test"], prediction)
        result["median_absolute_error"] = median_absolute_error(data["y_test"], prediction)
        result["r2_score"] = r2_score(data["y_test"], prediction)
        json_byte_object = json.dumps(result)

        # Save model evaluation results to S3
        date = datetime.now().strftime("%Y%m%d%H")
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")
        resource.Object(FIRST_BUCKET, f"{ROOT_FOLDER}/results/{date}.json").put(Body=json_byte_object)

        _LOG.info("Model training finished.")

    except Exception as e:
        _LOG.error(f"Error in model training: {str(e)}")
        raise e

# Function to save results
def save_results() -> None:
    _LOG.info("Success.")

# Define PythonOperator tasks in the DAG and their dependencies
task_init = PythonOperator(task_id="init", python_callable=init, dag=first_ml_dag)
task_get_data = PythonOperator(task_id="get_data_from_postgres", python_callable=get_data_from_postgres, dag=first_ml_dag)
task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=first_ml_dag)
task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=first_ml_dag)
task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=first_ml_dag)

# Define task dependencies in the DAG
task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results
