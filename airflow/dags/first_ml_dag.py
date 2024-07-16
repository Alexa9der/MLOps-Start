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


# import sys
# sys.path.append('data/config')
# from config import  _LOG ,DEFAULT_ARGS, first_ml_dag, FIRST_BUCKET, ROOT_FOLDER, RAW_DATA


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

DEFAULT_ARGS = {
    "owner": "Oleksandr Kanalosh",
    "email": ["oleksandrairflow@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5) 
}

first_ml_dag = DAG(
    dag_id="first_ml_dag",
    schedule_interval="0 1 * * *",  
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],  
    default_args=DEFAULT_ARGS
)

FIRST_BUCKET = "first-airflou-test"
ROOT_FOLDER = "First_action"
RAW_DATA = "First_action/california_housing.pkl"


FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"]

TARGET = "MedHouseVal"

def init() -> None:
    _LOG.info("Train pipeline started.")

def get_data_from_postgres() -> None:
   #TO-DO: Заполнить все шаги
    
    # Использовать созданный ранее PG connection
    pg_hook = PostgresHook("pd_connection")
    con = pg_hook.get_conn()
    
    # Прочитать все данные из таблицы california_housing
    data = pd.read_sql_query("SELECT * FROM california_housing", con)
   
    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")
    
    # Сохранить файл в формате pkl на S3
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(FIRST_BUCKET, RAW_DATA).put(Body=pickle_byte_obj)
    
    _LOG.info("Data download finished.")
    
def prepare_data() -> None:
    data = None 
    
    try:
        # Использовать созданный ранее S3 connection
        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(key=RAW_DATA, bucket_name=FIRST_BUCKET)

        data = pd.read_pickle(file)

        # Проверяем, что FEATURES и TARGET доступны в данных
        if any(feature not in data.columns for feature in FEATURES) or TARGET not in data.columns:
            raise ValueError(f"Features ({FEATURES}) or Target ({TARGET}) columns are missing in the dataset.")

        X, y = data[FEATURES].copy(), data[TARGET].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

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
    
    
def train_model() -> None:
    #TO-DO: Заполнить все шаги
    try:
        # Использовать созданный ранее S3 connection
        s3_hook = S3Hook("s3_connection")
    
        names = "X_train", "X_test", "y_train", "y_test"
        data = {}
        for name in names:
            file = s3_hook.download_file(key=f"{ROOT_FOLDER}/prepared_datasets/{name}.pkl", bucket_name=FIRST_BUCKET )
            data[name] = pd.read_pickle(file).copy()
        
        # Обучить модель
        model = RandomForestRegressor()
        model.fit(data["X_train"],data["y_train"])
        prediction = model.predict(data["X_test"])
        
        # Посчитать метрики
        result = {}
        result["mean_squared_error"] = mean_squared_error(data["y_test"], prediction)
        result["median_absolute_error"] = median_absolute_error(data["y_test"], prediction)
        result["r2_score"] = r2_score(data["y_test"], prediction)
        json_byte_object = json.dumps(result)
    
        # Сохранить результат на S3
        date = datetime.now().strftime("%Y%m%d%H")
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")
    
        resource.Object(FIRST_BUCKET, f"{ROOT_FOLDER}/results/{date}.json").put(Body=json_byte_object)
        
        _LOG.info("Model training finished.")

    except Exception as e:
        _LOG.error(f"Error in data preparation: {str(e)}")
        raise e
    

def save_results() -> None:
    _LOG.info("Success.")


task_init =  PythonOperator(task_id="init", python_callable=init, dag=first_ml_dag)

task_get_data = PythonOperator(task_id="get_data_from_postgres", python_callable=get_data_from_postgres, dag=first_ml_dag)

task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=first_ml_dag)

task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=first_ml_dag)

task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=first_ml_dag)



task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results