from datetime import timedelta
from typing import NoReturn

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner": "Oleksandr Kanalosh",
    "email": ["oleksandrairflow@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5) 
}

dag = DAG(
    dag_id="first_dag",
    schedule_interval="0 1 * * *",  
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],  
    default_args=DEFAULT_ARGS
)

def init() -> NoReturn:
    print("Hello World")

task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_init  
