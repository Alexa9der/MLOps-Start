from datetime import timedelta
import logging
from airflow.models import DAG
from airflow.utils.dates import days_ago

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