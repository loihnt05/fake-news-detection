# Dockerfile.app
FROM python:3.10-slim

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết (để build psycopg2, numpy...)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt uv để quản lý package siêu tốc
RUN pip install uv

# Copy file quy định thư viện
COPY pyproject.toml .

# Cài đặt thư viện Python (Backend + AI + Dashboard)
# Tạo file requirements.txt tạm từ pyproject.toml
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt --system

# Copy toàn bộ code dự án vào container
COPY . .

# Mặc định chạy bash (sẽ bị ghi đè bởi command trong docker-compose)
CMD ["bash"]