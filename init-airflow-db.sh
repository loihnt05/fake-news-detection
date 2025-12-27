#!/bin/bash
set -e

# Script này sẽ chạy bên trong container Postgres khi khởi tạo lần đầu
# Nó dùng user root của postgres để tạo thêm database tên là 'airflow'

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE airflow;
    GRANT ALL PRIVILEGES ON DATABASE airflow TO "$POSTGRES_USER";
EOSQL