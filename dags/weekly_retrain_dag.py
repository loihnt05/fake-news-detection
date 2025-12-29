from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

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
            'POSTGRES_USER': os.getenv('POSTGRES_USER', 'postgres'),
            'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'POSTGRES_DB': os.getenv('POSTGRES_DB', 'fake_news_db'),
        }
    )

    # Task 2: Gọi API Backend để reload model
    reload_api_task = BashOperator(
        task_id='trigger_backend_reload',
        bash_command='curl -X POST "http://backend:8000/api/internal/reload-model?secret_key=SUPER_SECRET_AIRFLOW_KEY" -H "Content-Type: application/json" || true',
    )

    train_task >> reload_api_task