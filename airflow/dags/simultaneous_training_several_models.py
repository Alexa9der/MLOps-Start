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

# Basic DAG parameters
DEFAULT_ARGS = {
    "owner": "Oleksandr Kanalosh",
    "email": ["oleksandrairflow@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5) 
}

# Basic variables
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


def init() -> None:
    _LOG.info("Initialization metrics.")
    
    metrics = {}
    metrics["start_timstamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics

def get_data(**kwargs) -> Dict[str, Any]:

    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "init", )

    _LOG.info("Retrieving data from Postgres and storing it in AWS S3.")
    return metrics

def prepare_data(**kwargs) -> Dict[str, Any]:
    _LOG.info("Beginning pre-processing of data.")
    
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "get_data")
    
    _LOG.info("End of data preprocessing.")
    return metrics

def train_model(*args, **kwargs) -> Dict[str, Any]:
    _LOG.info("Beginning train model.")
    
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "prepare_data")

    m_name = kwargs["model"]
    
    s3_hook = S3Hook("s3_connection")
    
    names = "X_train", "X_test", "y_train", "y_test"
    data = {}
    for name in names:
        file = s3_hook.download_file(key=f"{ROOT_FOLDER}/prepared_datasets/{name}.pkl", bucket_name=FIRST_BUCKET )
        data[name] = pd.read_pickle(file).copy()

    model = models[m_name]
    model.fit(data["X_train"],data["y_train"])
    metrics["train_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    prediction = model.predict(data["X_test"])
    metrics["mean_squared_error"] = mean_squared_error(data["y_test"], prediction)
    metrics["median_absolute_error"] = median_absolute_error(data["y_test"], prediction)
    metrics["r2_score"] = r2_score(data["y_test"], prediction)

    _LOG.info("End train model.")
    
    return metrics

def save_results(**kwargs) -> None:

    _LOG.info("Beginning save metrics.")
    
    ti = kwargs["ti"]

    metrics_rf = ti.xcom_pull(task_ids="train_model_rf")
    metrics_lr = ti.xcom_pull(task_ids="train_model_lr")
    metrics_hgb = ti.xcom_pull(task_ids="train_model_hgb")

    metrics = {
        "rf": metrics_rf,
        "lr": metrics_lr,
        "hgb": metrics_hgb
        }
    
    metrics["end_tiemestamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    json_byte_object = json.dumps(metrics)

    date = datetime.now().strftime("%Y%m%d%H")
    
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    
    resource.Object(FIRST_BUCKET, f"{ROOT_FOLDER}/results/metrics_several_models_{date}.json").put(Body=json_byte_object)

    _LOG.info("End save metrics.")


linearly_several_models = DAG(
    dag_id="Train_several_models",
    schedule_interval="0 1 1 * *",  
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],  
    default_args=DEFAULT_ARGS
)

with linearly_several_models:
    # Initializing the task to perform initialization steps
    task_init = PythonOperator(task_id="init", python_callable=init, dag=linearly_several_models, provide_context=True)
    
    # Fetching data task
    task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=linearly_several_models, provide_context=True)
    
    # Preparing data for model training
    task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=linearly_several_models, provide_context=True)
    
    # Training Random Forest model
    train_model_rf = PythonOperator(task_id="train_model_rf", python_callable=train_model, dag=linearly_several_models, op_kwargs={"model":"rf"}, provide_context=True)
    
    # Training Linear Regression model
    train_model_lr = PythonOperator(task_id="train_model_lr", python_callable=train_model, dag=linearly_several_models, op_kwargs={"model":"lr"}, provide_context=True)
    
    # Training Histogram Gradient Boosting model
    train_model_hgb = PythonOperator(task_id="train_model_hgb", python_callable=train_model, dag=linearly_several_models, op_kwargs={"model":"hgb"}, provide_context=True)
    
    # Saving results after model training
    task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=linearly_several_models, provide_context=True)

# Defining the sequence of tasks in the DAG
task_init >> task_get_data >> task_prepare_data >> [train_model_rf, train_model_lr, train_model_hgb] >> task_save_results





