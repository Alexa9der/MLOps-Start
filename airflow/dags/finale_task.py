"""
  - BUCKET заменить на свой;
  - EXPERIMENT_NAME и DAG_ID оставить как есть (ссылками на переменную NAME);
  - имена коннекторов: pg_connection и s3_connection;
  - данные должны читаться из таблицы с названием california_housing;
  - данные на S3 должны лежать в папках {NAME}/datasets/ и {NAME}/results/.
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

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

# Получаем соединение с S3 из Airflow Connections
conn = BaseHook.get_connection("s3_connection")

# Устанавливаем переменные окружения для доступа к AWS S3
os.environ["AWS_ACCESS_KEY_ID"] = conn.login
os.environ["AWS_SECRET_ACCESS_KEY"] = conn.password
os.environ["AWS_DEFAULT_REGION"] = conn.extra_dejson.get('region_name')
os.environ["MLFLOW_S3_ENDPOINT_URL"] = conn.extra_dejson.get('endpoint_url')

# Устанавливаем URI для MLflow трекинга
mlflow.set_tracking_uri("http://localhost:5000")

NAME = "Kanalosh_Oleksandr" # TO-DO: Вписать свой ник в телеграме
BUCKET = "mlops-start" # TO-DO: Вписать свой бакет

EXPERIMENT_NAME = NAME
DAG_ID = NAME

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"

models =   dict(zip(
       ["rf", "lr", "hgb"],
       [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]
)) # TO-DO: Создать словарь моделей


def init() -> Dict[str, Any]:
    print("Текущий рабочий каталог:", os.getcwd())

    _LOG.info("Initialization.")
    
    # TO-DO 1 metrics: В этом шаге собрать start_timestamp, run_id, experiment_name, experiment_id.
    metrics = {}
    metrics["start_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Start timestamp collected.")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        _LOG.info(f"Creating new experiment: {EXPERIMENT_NAME}")
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f"s3://{BUCKET}/mlflow")
    else:
        _LOG.info(f"Experiment {EXPERIMENT_NAME} already exists.")
        experiment_id = experiment.experiment_id
        
    mlflow.set_experiment(EXPERIMENT_NAME)

    metrics["experiment_name"] = EXPERIMENT_NAME
    metrics["experiment_id"] = experiment_id
    
    _LOG.info(f"Experiment set with ID: {experiment_id}")
    
    # TO-DO 3 mlflow: Создать parent run.
    with mlflow.start_run(experiment_id=experiment_id, run_name="Parent_Run") as parent_run:
        metrics['run_id'] = parent_run.info.run_id
        
    _LOG.info("Initialization end.")

    return metrics
    

def get_data_from_postgres(**kwargs) -> Dict[str, Any]:

    _LOG.info("Starting data extraction from PostgreSQL.")
    # TO-DO 1 metrics: В этом шаге собрать data_download_start, data_download_end.
    # your code here.
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")

    metrics["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    # TO-DO 2 connections: Создать коннекторы.
    # your code here.
    pg_hook = PostgresHook("pd_connection")
    con = pg_hook.get_conn()
    _LOG.info("Connected to PostgreSQL.")
    
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")
    _LOG.info("Connected to S3.")

    # TO-DO 3 Postgres: Прочитать данные.
    # your code here.
    data = pd.read_sql_query("SELECT * FROM california_housing", con)
    _LOG.info("Data read from PostgreSQL.")

    # TO-DO 4 Postgres: Сохранить данные на S3 в формате pickle в папку {NAME}/datasets/.
    file_name = f"{NAME}/datasets/california_housing.pkl"
    # your code here:
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, file_name).put(Body=pickle_byte_obj)
    _LOG.info("Data saved to S3.")
    
    metrics["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Data download end timestamp logged.")

    return metrics

def prepare_data(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать data_preparation_start, data_preparation_end.
    # your code here.
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="get_data_from_postgres")
    
    metrics["data_preparation_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    # TO-DO 2 connections: Создать коннекторы.
    # your code here.
    s3_hook = S3Hook("s3_connection")
    _LOG.info("Connected to S3.")
    
    # TO-DO 3 S3: Прочитать данные с S3.
    file_name = f"{NAME}/datasets/california_housing.pkl"
    # your code here.
    file = s3_hook.download_file(key=file_name, bucket_name=BUCKET)
    data = pd.read_pickle(file)
    X, y = data[FEATURES], data[TARGET]
    
    # TO-DO 4 Разделить данные на train/test.
    # your code here.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    
    # TO-DO 5 Сделать препроцессинг.
    # your code here.
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # TO-DO 6 Подготовить 4 обработанных датасета.
    # your code here.
    names = "X_train_fitted", "X_test_fitted", "y_train", "y_test"
    datas = X_train_fitted, X_test_fitted, y_train, y_test
    
    # Сохранить данные на S3 в папку {NAME}/datasets/.
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    for name, data in zip(names, datas):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET, f"{NAME}/datasets/{name}.pkl").put(Body=pickle_byte_obj)
    _LOG.info("Data download in S3.")

    metrics["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("End of data preprocessing.")
    
    return metrics


def train_mlflow_model(model: Any, name: str, X_train: np.array,
                       X_test: np.array, y_train: pd.Series,
                       y_test: pd.Series) -> None:

    # TO-DO 1: Обучить модель.
    # your code here
    model.fit(X_train, y_train)

    # TO-DO 2: Сделать predict.
    # your code here
    prediction = model.predict(X_test)

    # TO-DO 3: Сохранить результаты обучения с помощью MLFlow.
    # your code here
    # Получить описание данных
    signature = infer_signature(X_test, prediction)
    # Сохранить модель в артифактори
    model_info = mlflow.sklearn.log_model(model, name, signature=signature)
    # Сохранить метрики модели
    mlflow.evaluate(
        model_info.model_uri,
        data=X_test,
        targets=y_test.values,
        model_type="regressor",
        evaluators=["default"],
    )

def train_model(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать f"train_start_{model_name}" и f"train_end_{model_name}".
    # your code here.
    model_name = kwargs['model_name']
    _LOG.info(f"Training model: {model_name}")
    
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="prepare_data")
    metrics[f"train_start_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    # TO-DO 2 connections: Создать коннекторы.
    s3_hook = S3Hook("s3_connection")
    _LOG.info("Connected to S3.")

    # TO-DO 3 S3: Прочитать данные с S3 из папки {NAME}/datasets/.
    names = "X_train_fitted", "X_test_fitted", "y_train", "y_test"
    data = {}
    for name in names:
        file = s3_hook.download_file(key=f"{NAME}/datasets/{name}.pkl", bucket_name=BUCKET )
        data[name] = pd.read_pickle(file).copy()


    # TO-DO 4: Обучить модели и залогировать обучение с помощью MLFlow.
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
      
    with mlflow.start_run(run_name=f"{model_name}_Training", experiment_id=experiment_id, nested=True) as child_run:
        
        mlflow.set_tags({"mlflow.parentRunId": metrics['run_id']})
        train_mlflow_model(models[model_name], model_name, data["X_train_fitted"], data["X_test_fitted"], data["y_train"], data["y_test"]) 
        
        _LOG.info(f"Run completed for model: {model_name}")


    metrics[f"train_end_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info(f"Training for model {model_name} completed.")

    return metrics


def save_results(**kwargs) -> None:
    _LOG.info("Beginning save metrics.")
    _LOG.info(kwargs)
    
    # TO-DO 1 metrics: В этом шаге собрать end_timestamp.
    ti = kwargs["ti"]
    
    result = {}
    for model_name in  models.keys():
        model_metrics = ti.xcom_pull(task_ids=f"train_{model_name}")
        result.update(model_metrics)

    date = datetime.now().strftime("%Y_%m_%d_%H")
    result["end_timestamp"] = date
    json_byte_object = json.dumps(result)

    # TO-DO 2: сохранить результаты обучения на S3 в файл {NAME}/results/{date}.json.
    # your code here.
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")
    _LOG.info("Connected to S3.")

    file_name = f"{NAME}/results/{date}.json"
    resource.Object( BUCKET, file_name).put(Body=json_byte_object)
    
    _LOG.info("End save metrics.")



#################################### INIT DAG ####################################

default_args = {
    "owner": "Oleksandr Kanalosh",
    "email": ["oleksandrairflow@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5) 
    # TO-DO: Заполнить своими данными: настроить владельца и политику retries.
}

dag = DAG(
      # TO-DO: Заполнить остальными параметрами.
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

# TO-DO: Прописать архитектуру DAG'a.
task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results

