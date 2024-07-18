from datetime import timedelta
from typing import NoReturn

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Define default arguments for Airflow tasks
DEFAULT_ARGS = {
    "owner": "Oleksandr Kanalosh",
    "email": ["oleksandrairflow@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5) 
}

# Define the DAG object
dag = DAG(
    dag_id="first_dag",  # Unique identifier for the DAG
    schedule_interval="0 1 * * *",  # Schedule to run daily at 1:00 AM
    start_date=days_ago(2),  # Start 2 days ago
    catchup=False,  # Do not catch up with missed DAG runs
    tags=["mlops"],  # Tags to categorize the DAG
    default_args=DEFAULT_ARGS  # Default arguments for tasks in the DAG
)

# Define a Python function to initialize the task
def init() -> NoReturn:
    print("Hello World")  # Print statement as an example task action

# Define a PythonOperator task within the DAG
task_init = PythonOperator(
    task_id="init",  # Unique identifier for the task
    python_callable=init,  # Function to be called by the task
    dag=dag  # Assign the task to the created DAG
)

task_init  # This represents the task in Airflow

