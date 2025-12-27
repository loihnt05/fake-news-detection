from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import HttpOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'admin',
    'retries': 0,
}

with DAG(
    '2_weekly_model_retraining',
    default_args=default_args,
    description='Retrain model định kỳ dựa trên feedback',
    schedule_interval='0 0 * * 0', # Chạy 00:00 Chủ Nhật hàng tuần
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Chạy script Fine-tune
    # Sử dụng python trực tiếp (đã cài trong Dockerfile.airflow)
    train_task = BashOperator(
        task_id='finetune_model',
        bash_command='cd /opt/project && python model/retrain_pipeline.py',
        env={
            'POSTGRES_HOST': 'db',
            'POSTGRES_USER': '{{ var.value.POSTGRES_USER }}',
            'POSTGRES_PASSWORD': '{{ var.value.POSTGRES_PASSWORD }}',
            'POSTGRES_DB': '{{ var.value.POSTGRES_DB }}',
        }
    )

    # Task 2: Gọi API Backend để reload model (Webhook)
    # Cần tạo HTTP Connection trong Airflow UI trước:
    # Admin -> Connections -> Add -> 
    #   Conn Id: backend_api_connection
    #   Host: backend
    #   Port: 8000
    #   Schema: http
    reload_api_task = HttpOperator(
        task_id='trigger_backend_reload',
        http_conn_id='backend_api_connection',
        endpoint='/api/internal/reload-model?secret_key=SUPER_SECRET_AIRFLOW_KEY',
        method='POST',
        headers={"Content-Type": "application/json"},
    )

    train_task >> reload_api_task

    train_task >> reload_api_task