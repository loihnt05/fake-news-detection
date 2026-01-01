from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'admin',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    '1_daily_news_ingestion',
    default_args=default_args,
    description='Cào báo và cập nhật Knowledge Base hàng ngày',
    schedule_interval='0 6 * * *', # Chạy lúc 6:00 sáng mỗi ngày
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Chạy Crawler (Batch mode)
    # Uses batch_crawler.py which:
    # 1. Connects to Kafka using the correct Docker service name
    # 2. Sends test articles to verify connectivity
    # 3. Runs the scraper script
    crawl_task = BashOperator(
        task_id='run_crawler',
        bash_command='cd /opt/project && python crawler/batch_crawler.py',
        env={
            'KAFKA_SERVER': 'kafka:9093',  # Use Docker service name, not localhost
            'PYTHONPATH': '/opt/project',
        },
    )

    # Task 2: Rebuild Knowledge Base sau khi crawl xong
    # Script này sẽ tái xử lý các bài báo mới, trích xuất claims và tạo embeddings
    rebuild_kb_task = BashOperator(
        task_id='rebuild_knowledge_base',
        bash_command='cd /opt/project && python processor/rebuild_knowledge_base.py',
    )

    crawl_task >> rebuild_kb_task