import io
import json
import numpy as np
import pandas as pd
import pickle
import logging
from typing import NoReturn, Literal, Dict, Any

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

# Основные параметры DAG
DEFAULT_ARGS = {
    "owner": "Oleksandr Kanalosh",
    "email": ["oleksandrairflow@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5) 
}

# Базовые переменные
FIRST_BUCKET = "first-airflou-test"
ROOT_FOLDER = "First_action"
RAW_DATA = "First_action/california_housing.pkl"

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"

models = dict(zip(
       ["rf", "lr", "hgb"],
       [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]
))



# Function to create a Directed Acyclic Graph (DAG) for training multiple models
def create_dag(dag_id: str, m_name=Literal["rf", "lr", "hgb"]):
    
    # Function to initialize metrics
    def init() -> None:
        _LOG.info("Initialization metrics.")
        
        # Initialize metrics dictionary with model name and start timestamp
        metrics = {}
        metrics["model"] = m_name
        metrics["start_timstamp"] = datetime.now().strftime("%Y%m%d %H:%M")
        
        return metrics

    # Function to retrieve data
    def get_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="init")
        
        _LOG.info("Retrieving data from Postgres and storing it in AWS S3.")
        return metrics

    # Function to preprocess data
    def prepare_data(**kwargs) -> Dict[str, Any]:
        _LOG.info("Beginning pre-processing of data.")
        
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="get_data")
        
        _LOG.info("End of data preprocessing.")
        return metrics

    # Function to train a model
    def train_model(**kwargs) -> Dict[str, Any]:
        _LOG.info("Beginning train model.")
        
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="prepare_data")

        m_name = metrics["model"]
            
        s3_hook = S3Hook("s3_connection")
        
        # Downloading preprocessed data from S3
        names = "X_train", "X_test", "y_train", "y_test"
        data = {}
        for name in names:
            file = s3_hook.download_file(key=f"{ROOT_FOLDER}/prepared_datasets/{name}.pkl", bucket_name=FIRST_BUCKET)
            data[name] = pd.read_pickle(file).copy()

        # Training the selected model
        model = models[m_name]
        model.fit(data["X_train"], data["y_train"])
        
        # Evaluating model performance
        metrics["train_end"] = datetime.now().strftime("%Y%m%d %H:%M")
        prediction = model.predict(data["X_test"])
        metrics["mean_squared_error"] = mean_squared_error(data["y_test"], prediction)
        metrics["median_absolute_error"] = median_absolute_error(data["y_test"], prediction)
        metrics["r2_score"] = r2_score(data["y_test"], prediction)

        _LOG.info("End train model.")
        
        return metrics

    # Function to save results
    def save_results(**kwargs) -> None:
        _LOG.info("Beginning save metrics.")
        
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="train_model")
        metrics["end_tiemestamp"] = datetime.now().strftime("%Y%m%d %H:%M")

        # Convert metrics to JSON format
        json_byte_object = json.dumps(metrics)
    
        # Save results to S3
        date = datetime.now().strftime("%Y%m%d%H")
        s3_hook = S3Hook("s3_connection")
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")
        
        model = metrics["model"]
        resource.Object(FIRST_BUCKET, f"{ROOT_FOLDER}/results/{model}_{date}.json").put(Body=json_byte_object)

        _LOG.info("End save metrics.")

    # Creating the DAG object
    linearly_several_models = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",  # Run daily at 1:00 AM
        start_date=days_ago(2),  # Start 2 days ago
        catchup=False,  # Don't backfill past DAG runs
        tags=["mlops"],  # Tags for identifying the DAG's purpose
        default_args=DEFAULT_ARGS  # Default arguments for tasks in the DAG
    )

    # Defining tasks and their dependencies in the DAG
    with linearly_several_models:
        task_init = PythonOperator(task_id="init", python_callable=init, dag=linearly_several_models, provide_context=True)
        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=linearly_several_models, provide_context=True)
        task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=linearly_several_models, provide_context=True)
        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=linearly_several_models, provide_context=True)
        task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=linearly_several_models, provide_context=True)

    # Defining the sequence of tasks in the DAG
    task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


# Creating DAGs for each model in the models dictionary
for model_name in models.keys():
    # Calling create_dag function to create a DAG for training the current model
    create_dag(dag_id=f"{model_name}_train", m_name=model_name)











